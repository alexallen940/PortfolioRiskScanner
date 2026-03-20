"""
To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""

import json
import os
from flask import send_from_directory, request, jsonify
from models import db, Article, RiskData
from services.recommender import get_stock_recommendations, get_recommendation_desc
from services.risk import get_portfolio_risk_score, get_portfolio_risk_types

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

        if not portfolio:
            return jsonify({"error": "Portfolio not provided"})

        recommendations = get_stock_recommendations(portfolio)

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
