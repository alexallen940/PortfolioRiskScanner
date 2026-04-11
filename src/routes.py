"""
To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""

import json
import os
from flask import send_from_directory, request, jsonify
from models import db, Article, RiskData
from services.recommender import get_stock_recommendations, get_recommendation_desc
from services.risk import get_portfolio_risk_score, get_portfolio_risk_types
from sklearn.feature_extraction.text import TfidfVectorizer


# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
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
        data = request.get_json()
        portfolio = data.get("portfolio", [])
        desired_characteristics = data.get("desired_characteristics", "").lower().strip()

        if not portfolio:
            return jsonify({"error": "Portfolio not provided"})
        
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

        recommendations = get_stock_recommendations(
            portfolio,
            desired_characteristics=desired_characteristics,
            vectorizer=vectorizer,
        )

        return jsonify({"recommendations": recommendations})

    @app.route("/api/portfolio/risk-score", methods=["POST"])
    def portfolio_risk_score():
        data = request.get_json()
        portfolio = data.get("portfolio", [])

        if not portfolio:
            return jsonify({"error": "Portfolio not provided"})

        risk_score = get_portfolio_risk_score(portfolio)
        return jsonify({"risk_score": risk_score})

    @app.route("/api/portfolio/risk-types", methods=["POST"])
    def portfolio_risk_types():
        data = request.get_json()
        portfolio = data.get("portfolio", [])

        if not portfolio:
            return jsonify({"error": "Portfolio not provided"})

        risk_types = get_portfolio_risk_types(portfolio)
        return jsonify({"risk_types": risk_types})

    @app.route("/api/portfolio/recommendation-description", methods=["POST"])
    def recommendation_description():
        data = request.get_json()
        ticker = data.get("ticker", "").strip()

        if not ticker:
            return jsonify({"error": "Ticker not provided"})

        description = get_recommendation_desc(ticker)
        return jsonify({"ticker": ticker.upper(), "description": description})

    # @app.route("/api/episodes")
    # def episodes_search():
    #     text = request.args.get("title", "")
    #     return jsonify(json_search(text))

    if USE_LLM:
        from llm_routes import register_chat_route

        # register_chat_route(app, json_search)
