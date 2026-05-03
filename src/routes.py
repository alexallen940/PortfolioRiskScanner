"""
To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""

import csv
import difflib
import os
from flask import jsonify, request, send_from_directory
from services.recommender import (
    get_recommendation_desc,
    get_stock_recommendations,
    get_ticker_sentiment_summary,
)
from services.risk import (
    get_portfolio_risk_breakdown,
    get_portfolio_risk_score,
    get_portfolio_risk_types,
)

# ── AI toggle ────────────────────────────────────────────────────────────────
# USE_LLM = False
USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_WEIGHT_MAP = {"low": 110, "medium": 150, "high": 200}
_DEFAULT_QUERY_WEIGHT_LEVEL = "medium"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
constituents_path = os.path.join(BASE_DIR, "data", "constituents.csv")


def _load_sp500_tickers():
    tickers = set()
    if not os.path.exists(constituents_path):
        return tickers

    with open(constituents_path, "r") as file:
        for row in csv.DictReader(file):
            symbol = (row.get("Symbol") or row.get("ticker") or row.get("symbol") or "").strip().upper()
            if symbol:
                tickers.add(symbol)

    return tickers


SP500_TICKERS = _load_sp500_tickers()


def _normalize_portfolio(payload_portfolio):
    if not isinstance(payload_portfolio, list):
        return []

    normalized = []
    seen = set()
    for token in payload_portfolio:
        if not isinstance(token, str):
            continue
        ticker = token.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            normalized.append(ticker)

    return normalized


def _portfolio_validation_error(portfolio):
    if not portfolio:
        return jsonify({"error": "Portfolio not provided"}), 400

    if not SP500_TICKERS:
        return None

    invalid = [ticker for ticker in portfolio if ticker not in SP500_TICKERS]
    if not invalid:
        return None

    suggestions = {}
    sorted_tickers = sorted(SP500_TICKERS)
    for ticker in invalid:
        matches = difflib.get_close_matches(ticker, sorted_tickers, n=1, cutoff=0.7)
        if matches:
            suggestions[ticker] = matches[0]

    if suggestions:
        suggestion_text = "; ".join(f"{ticker} -> {match}" for ticker, match in suggestions.items())
        error_message = (
            f"Portfolio contains tickers outside the S&P 500: {', '.join(invalid)}. "
            f"Did you mean: {suggestion_text}?"
        )
    else:
        error_message = f"Portfolio contains tickers outside the S&P 500: {', '.join(invalid)}"

    return (
        jsonify({"error": error_message, "invalid_tickers": invalid, "suggestions": suggestions}),
        400,
    )


def _portfolio_from_request():
    data = request.get_json() or {}
    portfolio = _normalize_portfolio(data.get("portfolio", []))
    return portfolio, data, _portfolio_validation_error(portfolio)


def register_routes(app):
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/config", methods=["GET"])
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/portfolio/recommendations", methods=["POST"])
    def recommend():
        portfolio, data, err = _portfolio_from_request()
        if err:
            return err

        desired_characteristics = data.get("desired_characteristics", "").lower().strip()
        query_weight_level = (data.get("query_weight_level") or _DEFAULT_QUERY_WEIGHT_LEVEL).strip().lower()
        text_weight = _QUERY_WEIGHT_MAP.get(query_weight_level, _QUERY_WEIGHT_MAP[_DEFAULT_QUERY_WEIGHT_LEVEL])
        normalized_level = (
            query_weight_level if query_weight_level in _QUERY_WEIGHT_MAP else _DEFAULT_QUERY_WEIGHT_LEVEL
        )

        try:
            return jsonify(
                get_stock_recommendations(
                    portfolio,
                    desired_characteristics=desired_characteristics,
                    text_weight=text_weight,
                    text_weight_level=normalized_level,
                )
            )
        except Exception:
            return (
                jsonify(
                    {
                        "error": (
                            "Recommendation engine is warming up. "
                            "Please retry in a few seconds."
                        )
                    }
                ),
                503,
            )

    @app.route("/api/portfolio/risk-score", methods=["POST"])
    def portfolio_risk_score():
        portfolio, _, err = _portfolio_from_request()
        if err:
            return err
        return jsonify(
            {
                "risk_score": get_portfolio_risk_score(portfolio),
                "risk_breakdown": get_portfolio_risk_breakdown(portfolio),
            }
        )

    @app.route("/api/portfolio/risk-types", methods=["POST"])
    def portfolio_risk_types():
        portfolio, _, err = _portfolio_from_request()
        if err:
            return err
        return jsonify({"risk_types": get_portfolio_risk_types(portfolio)})

    @app.route("/api/portfolio/recommendation-description", methods=["POST"])
    def recommendation_description():
        data = request.get_json() or {}
        ticker = data.get("ticker", "").strip()
        use_llm = data.get("use_llm", False)

        if not ticker:
            return jsonify({"error": "Ticker not provided"})

        result = get_recommendation_desc(ticker, use_llm=use_llm)
        return jsonify(
            {
                "ticker": ticker.upper(),
                "description": result["bullets"],
                "description_details": result["details"],
            }
        )
        
    @app.route("/api/portfolio/recommendation-sentiment", methods=["POST"])
    def recommendation_sentiment():
        data = request.get_json() or {}
        ticker = data.get("ticker", "").strip()
        if not ticker:
            return jsonify({"error": "Ticker not provided"}), 400
        return jsonify(get_ticker_sentiment_summary(ticker))

    if USE_LLM:
        from llm_routes import register_llm_routes

        register_llm_routes(app)
