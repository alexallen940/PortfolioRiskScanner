"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds a POST /api/chat endpoint that performs LLM-driven RAG.

Setup:
    1. Add SPARK_API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""

import json
import os
import re
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient
from services.llm_services import _get_client, get_ticker_summary, get_risk_signals_for_tickers, get_ai_ticker_ranking


def tickers_summary_decision():

    data = request.get_json() or {}
    tickers = data.get("tickers", [])
    positive_bias = data.get("positive_bias", False)

    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    result = get_ticker_summary(tickers, _get_client(), positive_bias)
    return jsonify(result)


def tickers_risk_signals_decision():
    data = request.get_json() or {}
    recommended_tickers = data.get("tickers", [])

    if not recommended_tickers:
        return jsonify({"error": "No tickers provided"}), 400

    result = get_risk_signals_for_tickers(recommended_tickers, _get_client())
    return jsonify(result)


def ai_ticker_ranking_decision():
    data = request.get_json() or {}
    tickers = data.get("tickers", [])
    free_text_query = data.get("free_text_query", None)

    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    result = get_ai_ticker_ranking(tickers, _get_client(), free_text_query)
    return jsonify(result)


def register_ai_ticker_ranking_route(app):
    @app.route("/api/portfolio/ai-ticker-ranking", methods=["POST"])
    def ai_ticker_ranking_route():
        return ai_ticker_ranking_decision()


def register_tickers_summary_route(app):
    @app.route("/api/portfolio/recommendations-summary", methods=["POST"])
    def tickers_summary_route():
        return tickers_summary_decision()


def register_tickers_risk_signals_route(app):
    @app.route("/api/portfolio/recommendations-risk-signals", methods=["POST"])
    def tickers_risk_signals_route():
        return tickers_risk_signals_decision()

