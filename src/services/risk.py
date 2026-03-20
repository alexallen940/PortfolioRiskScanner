from collections import defaultdict
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from models import Article, RiskData
import numpy as np
from nltk.metrics.distance import edit_distance

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_directory))


def get_portfolio_risk_score(user_portfolio):
    scores = []
    for ticker in user_portfolio:
        # double check the proper formatting of the ticker
        ticker_formatted = ticker.strip().upper()

        # obtain the first row that matches the ticker
        risk_qry = RiskData.query.filter_by(ticker=ticker_formatted).first()
        if risk_qry:
            scores.append(risk_qry.risk_score_1_10)

    # if none of the portfolio stocks are in the S&P 500, return 5.0 risk score
    if not scores:
        return 5.0
    return round(sum(scores) / len(scores), 2)


def get_portfolio_risk_types(
    user_portfolio,
    top_k=5,
    vectorizer=TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3), min_df=5),
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
    ticker_docs = {ticker: " ".join(texts).replace('"', "") for ticker, texts in ticker_docs.items()}

    portfolio_texts = []

    for ticker in user_portfolio:
        if ticker in ticker_docs:
            portfolio_texts.append(ticker_docs[ticker])

    portfolio_doc = " ".join(portfolio_texts).replace('"', "")

    portfolio_vector = vectorizer.fit_transform([portfolio_doc]).flatten()

    risk_word_bank_file_path = os.path.join(project_root, "data", "risk_word_bank.json")

    with open(risk_word_bank_file_path, "r") as file:
        risk_word_bank = json.load(file)

    feature_names = vectorizer.get_feature_names_out()

    # indices of the most frequent features in the portfolio vector in descending order
    feature_freq_ranking_indices = np.argsort(portfolio_vector)[::-1]

    res = []

    for _, i in enumerate(feature_freq_ranking_indices):
        feature = feature_names[i]

        for risk_type, risk_signals in risk_word_bank.items():
            for risk_signal in risk_signals:
                if edit_distance(feature, risk_signal) <= 10:
                    res.append(risk_type)

    return res[:top_k]
