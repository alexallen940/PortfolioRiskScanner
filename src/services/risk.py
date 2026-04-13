from collections import defaultdict
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from models import Article, RiskData
import numpy as np
from nltk.metrics.distance import edit_distance

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_directory))



def get_portfolio_risk_score(user_portfolio):
    breakdown = get_portfolio_risk_breakdown(user_portfolio)
    if not breakdown:
        return 5.0
    return breakdown["final_score"]


def get_portfolio_risk_breakdown(user_portfolio):
    matched_rows = []
    for ticker in user_portfolio:
        ticker_formatted = ticker.strip().upper()
        risk_qry = RiskData.query.filter_by(ticker=ticker_formatted).first()
        if risk_qry:
            matched_rows.append(risk_qry)

    if not matched_rows:
        return None

    annualized_volatility = float(np.mean([row.annualized_volatility for row in matched_rows]))
    max_drawdown_abs = float(np.mean([abs(row.max_drawdown) for row in matched_rows]))
    var_95_abs = float(np.mean([abs(row.var_95) for row in matched_rows]))
    downside_volatility = float(np.mean([row.downside_volatility for row in matched_rows]))
    avg_daily_volume = float(np.mean([row.avg_daily_volume for row in matched_rows]))

    weighted_volatility = 0.30 * annualized_volatility
    weighted_drawdown = 0.25 * max_drawdown_abs
    weighted_var_95 = 0.20 * var_95_abs
    weighted_downside = 0.15 * downside_volatility
    weighted_volume_inverse = 0.10 * (1 / (avg_daily_volume + 1))

    raw_score = (
        weighted_volatility
        + weighted_drawdown
        + weighted_var_95
        + weighted_downside
        + weighted_volume_inverse
    )

    all_rows = RiskData.query.all()
    min_raw_score = min((float(row.raw_risk_score) for row in all_rows), default=0.0)
    max_raw_score = max((float(row.raw_risk_score) for row in all_rows), default=0.0)

    denominator = max_raw_score - min_raw_score
    if denominator > 0:
        normalized_score = 1 + 9 * ((raw_score - min_raw_score) / denominator)
    else:
        normalized_score = 5.0

    return {
        "weights": {
            "annualized_volatility": 0.30,
            "max_drawdown": 0.25,
            "var_95": 0.20,
            "downside_volatility": 0.15,
            "avg_daily_volume_inverse": 0.10,
        },
        "components": {
            "annualized_volatility": annualized_volatility,
            "max_drawdown_abs": max_drawdown_abs,
            "var_95_abs": var_95_abs,
            "downside_volatility": downside_volatility,
            "avg_daily_volume": avg_daily_volume,
            "avg_daily_volume_inverse": 1 / (avg_daily_volume + 1),
        },
        "weighted_components": {
            "annualized_volatility": weighted_volatility,
            "max_drawdown": weighted_drawdown,
            "var_95": weighted_var_95,
            "downside_volatility": weighted_downside,
            "avg_daily_volume_inverse": weighted_volume_inverse,
        },
        "raw_score": raw_score,
        "min_raw_score": min_raw_score,
        "max_raw_score": max_raw_score,
        "normalized_score": normalized_score,
        "final_score": round(normalized_score, 2),
        "matched_tickers": [row.ticker for row in matched_rows],
    }


def get_portfolio_risk_types(
    user_portfolio,
    top_k=5,
    vectorizer=TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3), min_df=1),
):
    ticker_docs = {}
    articles = Article.query.all()

    for article in articles:
        # combine headline and summary
        text = f"{article.headline} {article.summary}"

        if article.ticker not in ticker_docs:
            ticker_docs[article.ticker] = []

        # add the article text under that ticker
        ticker_docs[article.ticker].append(text)

    # convert lists of article texts into one big doc per ticker
    ticker_docs = {
        ticker: " ".join(texts).replace('"', "").replace("-", " ").lower() for ticker, texts in ticker_docs.items()
    }

    portfolio_texts = []

    for ticker in user_portfolio:
        if ticker in ticker_docs:
            portfolio_texts.append(ticker_docs[ticker])

    if not portfolio_texts:
        return []

    portfolio_doc = " ".join(portfolio_texts).replace('"', "").replace("-", " ").lower()

    # try:
    #     portfolio_vector = vectorizer.fit_transform([portfolio_doc]).toarray().flatten()
    # except ValueError:
    #     return []

    risk_word_bank_file_path = os.path.join(project_root, "data", "risk_word_bank.json")

    with open(risk_word_bank_file_path, "r") as file:
        risk_word_bank = json.load(file)

    # feature_names = vectorizer.get_feature_names_out()

    # # indices of the most frequent features in the portfolio vector in descending order
    # feature_freq_ranking_indices = np.argsort(portfolio_vector)[::-1]

    # res = []

    # for _, i in enumerate(feature_freq_ranking_indices):
    #     feature = feature_names[i]
    #     for risk_type, risk_signals in risk_word_bank.items():
    #         if risk_type in res:
    #             continue
    #         for risk_signal in risk_signals:
    #             if edit_distance(feature, risk_signal) <= 10:
    #                 if risk_type in res:
    #                     break
    #                 res.append(risk_type)

    # return res[:top_k]

    risk_type_counts = {}
    for risk_type in risk_word_bank:
        risk_type_counts[risk_type] = 0

    for risk_type, risk_signals in risk_word_bank.items():
        for risk_signal in risk_signals:
            pattern = rf"\b{re.escape(risk_signal)}\b"

            if re.search(pattern, portfolio_doc):
                risk_type_counts[risk_type] += 1

    ranked_risk_types = sorted(risk_type_counts.items(), key=lambda x: x[1], reverse=True)
    res = [risk_type for risk_type, count in ranked_risk_types if count > 0]
    return res[:top_k]
