"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds a POST /api/chat endpoint that performs LLM-driven RAG.

Setup:
  1. Add API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""

import json
import os
import re
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient
from models import Article

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
risk_word_bank_json_path = os.path.join(BASE_DIR, "data", "risk_word_bank.json")

with open(risk_word_bank_json_path, "r") as f:
    RISK_WORD_DICT = json.load(f)


def expand_stock_query(user_query, client):
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite user stock-screening queries for information retrieval.\n"
                "Expand the query into a concise search string that preserves the user's intent.\n"
                "Include related industry terms, business characteristics, and synonyms that would help retrieve relevant stocks.\n"
                "Do not mention specific stock tickers, company names, or names of public figures unless the user explicitly included them.\n"
                "Do not explain your reasoning.\n"
                "Return ONLY a short expanded query string."
            ),
        },
        {
            "role": "user",
            "content": f"User stock preference query: {user_query}",
        },
    ]

    response = client.chat(messages)
    content = (response.get("content") or "").strip()
    parsed = re.sub(r"\s+", " ", content)
    print("\n" + parsed + "\n")
    return parsed


def get_risk_signals_for_tickers(tickers, client):
    ticker_docs = {}
    articles = Article.query.filter(Article.ticker.in_(tickers)).all()

    for article in articles:
        ticker = article.ticker.upper().strip()
        if ticker not in ticker_docs:
            ticker_docs[ticker] = []
        ticker_docs[ticker].append(
            {"headline": article.headline, "text": f"{article.headline} {article.summary}".strip()}
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You will receive:\n"
                "1) A risk word dictionary where each key is a risk category and each value is a list of risk signal terms.\n"
                "2) A dictionary mapping stock tickers to a list of article objects.\n"
                "Each article object has:\n"
                "- headline\n"
                "- text\n\n"
                "Your task:\n"
                "For each ticker, return a nested JSON object where:\n"
                "- each ticker maps to risk categories,\n"
                "- each risk category maps to risk signal terms from the provided dictionary,\n"
                "- each risk signal term maps to an object with:\n"
                '  - "count": number of articles that semantically express that signal\n'
                '  - "article_indices": list of 0-based article indices supporting that signal\n\n'
                "Matching rules:\n"
                "- Match SEMANTICALLY, not only by exact keyword overlap.\n"
                "- A signal term counts if an article clearly expresses that idea, even using paraphrased, synonymous, or closely related wording.\n"
                "- Count the number of ARTICLES mentioning each signal, not raw word frequency.\n"
                "- Each article contributes at most 1 count per signal term.\n"
                "- A single article may count toward multiple signal terms if clearly supported.\n"
                "- Be conservative: if uncertain, do not count it.\n"
                "- Do NOT create new risk categories.\n"
                "- Do NOT create new risk signal terms.\n"
                "- Only use the exact risk categories and exact risk signal terms from the provided risk word dictionary.\n"
                "- Only include signal terms with count > 0.\n"
                "- Only include risk categories that contain at least one signal term with count > 0.\n"
                "- If a ticker has no detected signals, return an empty object for that ticker.\n"
                "- Make sure count equals the length of article_indices.\n\n"
                "Respond ONLY with valid JSON in this format:\n"
                "{\n"
                '  "AAPL": {\n'
                '    "operational risk": {\n'
                '      "supply disruption": {\n'
                '        "count": 3,\n'
                '        "article_indices": [0, 2, 5]\n'
                "      }\n"
                "    }\n"
                "  }\n"
                "}\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Risk word bank:\n{json.dumps(RISK_WORD_DICT, indent=2)}\n\n"
                f"Ticker articles:\n{json.dumps(ticker_docs, indent=2)}\n"
            ),
        },
    ]

    response = client.chat(messages)
    content = (response.get("content") or "").strip()

    try:
        parsed = json.loads(re.sub(r"```json|```", "", content).strip())
        # print("\n" + parsed + "\n")
        return {
            "signals": parsed,
            "ticker_docs": ticker_docs,
        }
    except (json.JSONDecodeError, TypeError):
        return {
            "signals": {},
            "ticker_docs": ticker_docs,
        }


def get_ticker_summary(tickers, client, positive_bias=False):
    ticker_docs = {}
    articles = Article.query.filter(Article.ticker.in_(tickers)).all()

    for article in articles:
        ticker = article.ticker.upper().strip()
        if ticker not in ticker_docs:
            ticker_docs[ticker] = []
        ticker_docs[ticker].append(
            {"headline": article.headline, "text": f"{article.headline} {article.summary}".strip()}
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You will receive:\n"
                "1) A risk word dictionary where each key is a risk category and each value is a list of risk signal terms.\n"
                "2) A dictionary mapping stock tickers to a list of article objects.\n"
                "3) A boolean flag indicating whether to use positive bias and if false, be harsh and find any risk signals you can.\n"
                "Each article object has:\n"
                "- headline\n"
                "- text\n\n"
                "Your task:\n"
                "For each ticker, return a nested JSON object where:\n"
                "- each ticker has a summary of the ticker's articles.\n\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Risk word bank:\n{json.dumps(RISK_WORD_DICT, indent=2)}\n\n"
                f"Ticker articles:\n{json.dumps(ticker_docs, indent=2)}\n"
                f"Positive bias: {positive_bias}\n"
            ),
        },
    ]

    response = client.chat(messages)
    content = (response.get("content") or "").strip()

    try:
        parsed = json.loads(re.sub(r"```json|```", "", content).strip())
        # print("\n" + parsed + "\n")
        return parsed
    except (json.JSONDecodeError, TypeError):
        return {}


def get_ai_ticker_ranking(tickers, client, free_text_query=""):

    ticker_docs = {}
    articles = Article.query.filter(Article.ticker.in_(tickers)).all()

    for article in articles:
        ticker = article.ticker.upper().strip()
        if ticker not in ticker_docs:
            ticker_docs[ticker] = []
        ticker_docs[ticker].append(
            {"headline": article.headline, "text": f"{article.headline} {article.summary}".strip()}
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You will receive:\n"
                "1) A dictionary mapping stock tickers to a list of article objects.\n"
                "Each article object has:\n"
                "- headline\n"
                "- text\n\n"
                "2) A list of stock tickers.\n"
                "3) An expanded free text query string describing the stock characteristics the user wants.\n\n"
                "Your task:\n"
                "Re-rank the input ticker list from most semantically aligned to least semantically aligned with the expanded free text query string.\n"
                "Use the article objects for each ticker as the primary evidence for determining alignment.\n"
                "The output should contain the same tickers as the input, reordered from best alignment to worst alignment with the expanded free text query string.\n\n"
                "Ranking rules:\n"
                "- Use SEMANTIC reasoning, not exact keyword overlap only.\n"
                "- Compare the expanded free text query string against the themes, risks, business characteristics, and signals present in each ticker's articles.\n"
                "- Rank higher the tickers whose article evidence is more aligned with the user's desired characteristics.\n"
                "- Rank lower the tickers whose article evidence is less aligned with the user's desired characteristics.\n"
                "- Be conservative and consistent in your ranking.\n"
                "- Do NOT add any new tickers.\n"
                "- Do NOT remove any tickers.\n"
                "- Do NOT return explanations.\n"
                "- Do NOT return scores, labels, or extra text.\n"
                "- Return ONLY a valid JSON array.\n\n"
                "Respond ONLY with valid JSON in this format:\n"
                "[\n"
                '  "PANW",\n'
                '  "QCOM",\n'
                '  "ACN",\n'
                '  "CSCO"\n'
                "]\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Expanded free text query:\n{json.dumps(free_text_query, indent=2)}\n\n"
                f"Tickers:\n{json.dumps(tickers, indent=2)}\n\n"
                f"Ticker articles:\n{json.dumps(ticker_docs, indent=2)}\n"
            ),
        },
    ]

    response = client.chat(messages)
    content = response.get("content")
    print(f"\n[RAW CONTENT]: {content!r}\n", flush=True)

    try:
        parsed = json.loads(re.sub(r"```json|```", "", content).strip())
        print(f"\n[PARSED]: {parsed}\n", flush=True)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError) as e:
        print(f"\n[PARSE FAILED]: {type(e).__name__}: {e}\n", flush=True)
        return []


def tickers_summary_decision():
    data = request.get_json() or {}
    tickers = data.get("tickers", [])
    positive_bias = data.get("positive_bias", False)

    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    api_key = os.getenv("SPARK_API_KEY")
    client = LLMClient(api_key=api_key)

    result = get_ticker_summary(tickers, client, positive_bias)
    return jsonify(result)


def tickers_risk_signals_decision():
    data = request.get_json() or {}
    recommended_tickers = data.get("tickers", [])

    if not recommended_tickers:
        return jsonify({"error": "No tickers provided"}), 400

    api_key = os.getenv("SPARK_API_KEY")
    client = LLMClient(api_key=api_key)

    result = get_risk_signals_for_tickers(recommended_tickers, client)
    return jsonify(result)


def ai_ticker_ranking_decision():
    data = request.get_json() or {}
    tickers = data.get("tickers", [])
    free_text_query = data.get("free_text_query", None)

    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    api_key = os.getenv("SPARK_API_KEY")
    client = LLMClient(api_key=api_key)

    result = get_ai_ticker_ranking(tickers, client, free_text_query)
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


def llm_search_decision(client, user_message):
    """Ask the LLM whether to search the DB and which word to use."""
    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a database of Keeping Up with the Kardashians episode titles, "
                "descriptions, and IMDB ratings. Search is by a single word in the episode title. "
                "Reply with exactly: YES followed by one space and ONE word to search (e.g. YES wedding), "
                "or NO if the question does not need episode data."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    response = client.chat(messages)
    content = (response.get("content") or "").strip().upper()
    logger.info(f"LLM search decision: {content}")
    if re.search(r"\bNO\b", content) and not re.search(r"\bYES\b", content):
        return False, None
    yes_match = re.search(r"\bYES\s+(\w+)", content)
    if yes_match:
        return True, yes_match.group(1).lower()
    if re.search(r"\bYES\b", content):
        return True, "Kardashian"
    return False, None


def register_chat_route(app, json_search):
    """Register the /api/chat SSE endpoint. Called from routes.py."""

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)
        use_search, search_term = llm_search_decision(client, user_message)

        if use_search:
            episodes = json_search(search_term or "Kardashian")
            context_text = (
                "\n\n---\n\n".join(
                    f"Title: {ep['title']}\nDescription: {ep['descr']}\nIMDB Rating: {ep['imdb_rating']}"
                    for ep in episodes
                )
                or "No matching episodes found."
            )
            messages = [
                {
                    "role": "system",
                    "content": "Answer questions about Keeping Up with the Kardashians using only the episode information provided.",
                },
                {"role": "user", "content": f"Episode information:\n\n{context_text}\n\nUser question: {user_message}"},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for Keeping Up with the Kardashians questions.",
                },
                {"role": "user", "content": user_message},
            ]

        def generate():
            if use_search and search_term:
                yield f"data: {json.dumps({'search_term': search_term})}\n\n"
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
