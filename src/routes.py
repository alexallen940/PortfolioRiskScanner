"""
To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""

import json
import os
import csv
import difflib
from flask import send_from_directory, request, jsonify
from models import db, Article, RiskData
from services.recommender import get_stock_recommendations, get_recommendation_desc, get_ticker_sentiment_summary
from services.risk import get_portfolio_risk_score, get_portfolio_risk_types, get_portfolio_risk_breakdown
from sklearn.feature_extraction.text import TfidfVectorizer


# ── AI toggle ────────────────────────────────────────────────────────────────
# USE_LLM = False
USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────


# def json_search(query):
#     if not query or not query.strip():
#         query = "Kardashian"
#     results = db.session.query(Episode, Review).join(
#         Review, Episode.id == Review.id
#     ).filter(
#         Episode.title.ilike(f'%{query}%')
#     ).all()
#     matches = []
#     for episode, review in results:
#         matches.append({
#             'title': episode.title,
#             'descr': episode.descr,
#             'imdb_rating': review.imdb_rating
#         })
#     return matches

# load the stop word bank
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
json_path = os.path.join(BASE_DIR, "data", "stop_words.json")

with open(json_path, "r") as f:
    STOP_WORDS = json.load(f)

constituents_path = os.path.join(BASE_DIR, "data", "constituents.csv")


def _load_sp500_tickers():
    tickers = set()

    if not os.path.exists(constituents_path):
        return tickers

    with open(constituents_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
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
            "Portfolio contains tickers outside the S&P 500: "
            f"{', '.join(invalid)}. Did you mean: {suggestion_text}?"
        )
    else:
        error_message = (
            "Portfolio contains tickers outside the S&P 500: "
            f"{', '.join(invalid)}"
        )

    return jsonify({"error": error_message, "invalid_tickers": invalid, "suggestions": suggestions}), 400


def register_routes(app):
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/config", methods=["GET"])
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/portfolio/recommendations", methods=["POST"])
    def recommend():
        data = request.get_json() or {}
        portfolio = _normalize_portfolio(data.get("portfolio", []))
        desired_characteristics = data.get("desired_characteristics", "").lower().strip()
        query_weight_level = (data.get("query_weight_level", "medium") or "medium").strip().lower()

        query_weight_map = {
            "low": 110,
            "medium": 150,
            "high": 200,
        }
        text_weight = query_weight_map.get(query_weight_level, 150)

        portfolio_error = _portfolio_validation_error(portfolio)
        if portfolio_error:
            return portfolio_error
        
        if not desired_characteristics:
            return jsonify({"error": "Free text query not provided"})

        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2), min_df=15, max_df=0.9)

        custom_stops = (
            list(vectorizer.get_stop_words())
            + [w.lower().strip() for w in STOP_WORDS["company_names"]]
            + [w.lower().strip() for w in STOP_WORDS["extra_words"]]
            + [w.lower().strip() for w in STOP_WORDS["people_names"]]
        )

        vectorizer = TfidfVectorizer(
            stop_words=custom_stops, max_features=5000, ngram_range=(1, 2), min_df=10, max_df=0.95
        )

        recommendation_result = get_stock_recommendations(
            portfolio,
            desired_characteristics=desired_characteristics,
            vectorizer=vectorizer,
            text_weight=text_weight,
            text_weight_level=query_weight_level if query_weight_level in query_weight_map else "medium",
        )

        return jsonify(recommendation_result)

    @app.route("/api/portfolio/risk-score", methods=["POST"])
    def portfolio_risk_score():
        data = request.get_json() or {}
        portfolio = _normalize_portfolio(data.get("portfolio", []))

        portfolio_error = _portfolio_validation_error(portfolio)
        if portfolio_error:
            return portfolio_error

        risk_score = get_portfolio_risk_score(portfolio)
        risk_breakdown = get_portfolio_risk_breakdown(portfolio)
        return jsonify({"risk_score": risk_score, "risk_breakdown": risk_breakdown})

    @app.route("/api/portfolio/risk-types", methods=["POST"])
    def portfolio_risk_types():
        data = request.get_json() or {}
        portfolio = _normalize_portfolio(data.get("portfolio", []))

        portfolio_error = _portfolio_validation_error(portfolio)
        if portfolio_error:
            return portfolio_error

        risk_types = get_portfolio_risk_types(portfolio)
        return jsonify({"risk_types": risk_types})

    @app.route("/api/portfolio/recommendation-description", methods=["POST"])
    def recommendation_description():
        data = request.get_json()
        ticker = data.get("ticker", "").strip()

        if not ticker:
            return jsonify({"error": "Ticker not provided"})

        result = get_recommendation_desc(ticker, use_llm=False)
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

        result = get_ticker_sentiment_summary(ticker)
        return jsonify(result)

    # @app.route("/api/episodes")
    # def episodes_search():
    #     text = request.args.get("title", "")
    #     return jsonify(json_search(text))

    if USE_LLM:
        from llm_routes import (
            register_ai_ticker_ranking_route,
            register_tickers_risk_signals_route,
            register_tickers_summary_route,
        )

        register_ai_ticker_ranking_route(app)
        register_tickers_risk_signals_route(app)
        register_tickers_summary_route(app)
