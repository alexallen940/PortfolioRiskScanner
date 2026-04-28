import json
import os
import re
import numpy as np
from models import RiskData


_RISK_BREAKDOWN_WEIGHTS = {
    "annualized_volatility": 0.30,
    "max_drawdown": 0.25,
    "var_95": 0.20,
    "downside_volatility": 0.15,
    "avg_daily_volume_inverse": 0.10,
}

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_directory))
risk_word_bank_json_path = os.path.join(project_root, "data", "risk_word_bank.json")

with open(risk_word_bank_json_path, "r") as f:
    RISK_KEYWORDS = json.load(f)


def get_portfolio_risk_score(user_portfolio):

    breakdown = get_portfolio_risk_breakdown(user_portfolio)
    if not breakdown:
        return 5.0
    return breakdown["final_score"]


def get_portfolio_risk_breakdown(user_portfolio):
    matched_rows = []
    for ticker in user_portfolio:
        risk_qry = RiskData.query.filter_by(ticker=ticker.strip().upper()).first()
        if risk_qry:
            matched_rows.append(risk_qry)

    if not matched_rows:
        return None

    annualized_volatility = float(np.mean([row.annualized_volatility for row in matched_rows]))
    max_drawdown_abs = float(np.mean([abs(row.max_drawdown) for row in matched_rows]))
    var_95_abs = float(np.mean([abs(row.var_95) for row in matched_rows]))
    downside_volatility = float(np.mean([row.downside_volatility for row in matched_rows]))
    avg_daily_volume = float(np.mean([row.avg_daily_volume for row in matched_rows]))
    avg_daily_volume_inverse = 1 / (avg_daily_volume + 1)

    weighted = {
        "annualized_volatility": _RISK_BREAKDOWN_WEIGHTS["annualized_volatility"] * annualized_volatility,
        "max_drawdown": _RISK_BREAKDOWN_WEIGHTS["max_drawdown"] * max_drawdown_abs,
        "var_95": _RISK_BREAKDOWN_WEIGHTS["var_95"] * var_95_abs,
        "downside_volatility": _RISK_BREAKDOWN_WEIGHTS["downside_volatility"] * downside_volatility,
        "avg_daily_volume_inverse": _RISK_BREAKDOWN_WEIGHTS["avg_daily_volume_inverse"] * avg_daily_volume_inverse,
    }
    raw_score = sum(weighted.values())

    from services.recommender import INDEX

    INDEX.ensure_built()
    min_raw_score = INDEX.min_raw_score
    max_raw_score = INDEX.max_raw_score

    denominator = max_raw_score - min_raw_score
    if denominator > 0:
        normalized_score = 1 + 9 * ((raw_score - min_raw_score) / denominator)
    else:
        normalized_score = 5.0

    return {
        "weights": dict(_RISK_BREAKDOWN_WEIGHTS),
        "components": {
            "annualized_volatility": annualized_volatility,
            "max_drawdown_abs": max_drawdown_abs,
            "var_95_abs": var_95_abs,
            "downside_volatility": downside_volatility,
            "avg_daily_volume": avg_daily_volume,
            "avg_daily_volume_inverse": avg_daily_volume_inverse,
        },
        "weighted_components": weighted,
        "raw_score": raw_score,
        "min_raw_score": min_raw_score,
        "max_raw_score": max_raw_score,
        "normalized_score": normalized_score,
        "final_score": round(normalized_score, 2),
        "matched_tickers": [row.ticker for row in matched_rows],
    }


def get_portfolio_risk_types(user_portfolio, top_k=5):
    from services.recommender import INDEX

    INDEX.ensure_built()
    ticker_docs = INDEX.ticker_docs

    portfolio_texts = [ticker_docs[t] for t in user_portfolio if t in ticker_docs]
    if not portfolio_texts:
        return []

    portfolio_doc = " ".join(portfolio_texts).replace('"', "").replace("-", " ").lower()

    risk_type_counts = {}
    for risk_type, risk_signals in RISK_KEYWORDS.items():
        count = 0
        for risk_signal in risk_signals:
            if re.search(rf"\b{re.escape(risk_signal)}\b", portfolio_doc):
                count += 1
        risk_type_counts[risk_type] = count

    ranked_risk_types = sorted(risk_type_counts.items(), key=lambda x: x[1], reverse=True)
    return [risk_type for risk_type, count in ranked_risk_types if count > 0][:top_k]
