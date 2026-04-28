"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds a POST /api/chat endpoint that performs LLM-driven RAG.

Setup:
    1. Add SPARK_API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""

from flask import jsonify, request

from services.llm_services import (
    _get_client,
    get_ai_overview,
    get_ai_ticker_ranking,
    get_risk_signals_for_tickers,
    get_ticker_summary,
)


def _tickers_from_request():
    data = request.get_json() or {}
    tickers = data.get("tickers", [])
    if not tickers:
        return data, tickers, (jsonify({"error": "No tickers provided"}), 400)
    return data, tickers, None


def tickers_summary_decision():
    data, tickers, err = _tickers_from_request()
    if err:
        return err
    positive_bias = data.get("positive_bias", False)
    return jsonify(get_ticker_summary(tickers, _get_client(), positive_bias))


def ai_overview_decision():
    data = request.get_json() or {}
    portfolio_tickers = data.get("portfolio_tickers", [])
    output_tickers = data.get("output_tickers", [])
    free_text_query = data.get("free_text_query", "")
    
    if not portfolio_tickers or not output_tickers:
        return jsonify({"error": "Both portfolio_tickers and output_tickers are required"}), 400

    try:
        overview = get_ai_overview(portfolio_tickers, output_tickers, _get_client(), free_text_query)
    except Exception as exc:
        if "429" in str(exc):
            overview = "AI overview temporarily unavailable due to rate limiting. Please retry in a moment."
        else:
            overview = "AI overview temporarily unavailable."

    return jsonify({"overview": overview})


def tickers_risk_signals_decision():
    _, tickers, err = _tickers_from_request()
    if err:
        return err
    return jsonify(get_risk_signals_for_tickers(tickers, _get_client()))


def ai_ticker_ranking_decision():
    data, tickers, err = _tickers_from_request()
    if err:
        return err
    free_text_query = data.get("free_text_query", None)
    return jsonify(get_ai_ticker_ranking(tickers, _get_client(), free_text_query))


_LLM_ROUTES = (
    ("/api/portfolio/ai-ticker-ranking", "ai_ticker_ranking_route", ai_ticker_ranking_decision),
    ("/api/portfolio/ai-overview", "ai_overview_dash_route", ai_overview_decision),
    ("/api/portfolio/ai_overview", "ai_overview_underscore_route", ai_overview_decision),
    ("/api/portfolio/recommendations-summary", "tickers_summary_route", tickers_summary_decision),
    ("/api/portfolio/recommendations-risk-signals", "tickers_risk_signals_route", tickers_risk_signals_decision),
)


def register_llm_routes(app):
    for url, endpoint, view in _LLM_ROUTES:
        app.add_url_rule(url, endpoint, view, methods=["POST"])
