import json
import os
import re
from infosci_spark_client import LLMClient


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RISK_WORD_BANK_PATH = os.path.join(BASE_DIR, "data", "risk_word_bank.json")

with open(RISK_WORD_BANK_PATH, "r") as f:
    RISK_WORD_DICT = json.load(f)

RISK_WORD_DICT_JSON = json.dumps(RISK_WORD_DICT)

_EXPAND_CACHE = {}
_EXPAND_CACHE_MAX = 100

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _build_ticker_docs(
    tickers,
    max_articles_per_ticker=30,
    max_summary_chars=200,
    max_headline_chars=150,
    summary_key="text",
):
    """Build the {ticker: [{headline, <summary_key>}, ...]} dict from cached INDEX data."""
    from services.recommender import INDEX

    INDEX.ensure_built()
    out = {}
    for ticker in tickers:
        normalized = ticker.upper().strip()
        articles = INDEX.ticker_article_rows.get(normalized, [])[:max_articles_per_ticker]
        out[normalized] = [
            {
                "headline": (article.headline or "")[:max_headline_chars].strip().lower(),
                summary_key: (article.summary or "")[:max_summary_chars].strip().lower(),
            }
            for article in articles
        ]
    return out


def _parse_json_response(content, expected="object"):
    """Return parsed JSON value or None if content can't be parsed.

    Tries direct parse first, then strips ``` fences, then falls back to
    extracting the first {...} or [...] match.
    """
    if not content:
        return None

    stripped = _JSON_FENCE_RE.sub("", content).strip()

    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    pattern = _JSON_OBJECT_RE if expected == "object" else _JSON_ARRAY_RE
    match = pattern.search(stripped)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None


_EXPAND_STOCK_QUERY_PROMPT = (
    "You rewrite user stock-screening queries for information retrieval.\n"
    "Expand the query into a concise search string that preserves the user's intent.\n"
    "Include related industry terms, business characteristics, and synonyms that would help retrieve relevant stocks.\n"
    "Do not mention specific stock tickers, company names, or names of public figures unless the user explicitly included them.\n"
    "If the user's query is empty, contains no clear descriptors, or carries no meaningful semantic content (e.g., gibberish, filler words, or vague phrases like 'stocks' or 'stuff'), return an empty string.\n"
    "Do not explain your reasoning.\n"
    "Return ONLY a short expanded query string, or an empty string if the input lacks meaningful content.\n"
)

_RISK_SIGNALS_TICKERS_PROMPT = (
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
    "- Count the number of ARTICLES matched to each signal.\n"
    "- Each article contributes at most 1 count per signal term.\n"
    "- A single article may count toward multiple signal terms.\n"
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
)

_TICKERS_SUMMARY_PROMPT = (
    "You will receive:\n"
    "1) A dictionary mapping stock tickers to a list of article objects.\n"
    "2) A boolean flag indicating whether to use positive bias and if false, be harsh and find any risk signals you can.\n"
    "Each article object has:\n"
    "- headline\n"
    "- text\n\n"
    "Your task:\n"
    "For each ticker, write a concise 3-5 sentence summary describing:\n"
    "- key risks\n"
    "- business context\n"
    "- overall outlook\n\n"
    "Guidelines:\n"
    "- Be specific and grounded in the articles\n"
    "- If positive_bias is true, emphasize strengths\n"
    "- If false, emphasize risk factors and weaknesses\n"
    "- Do NOT list bullet points\n"
    "- Use specific, concrete language grounded in the provided articles.\n"
    "- Keep summaries to 3-5 sentences max.\n"
    "- Do not repeat the same idea across sentences.\n"
    "- Do NOT return structured categories\n\n"
    "Output format:\n"
    "{\n"
    '  "AAPL": "summary text...",\n'
    '  "NVDA": "summary text..."\n'
    "}\n"
)

_AI_TICKER_RANKING_PROMPT = (
    "You will receive:\n"
    "1) ticker_docs, a dictionary mapping stock tickers to a list of article objects.\n"
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
)

_AI_OVERVIEW_PROMPT = (
    "You will receive:\n"
    "1) Portfolio ticker articles, a dictionary mapping portfolio stock tickers to a list of article objects.\n"
    "Each article object has:\n"
    "- headline\n"
    "- text\n\n"
    "2) Output ticker articles, a dictionary mapping output stock tickers to a list of article objects.\n"
    "Each article object has:\n"
    "- headline\n"
    "- text\n\n"
    "3) An expanded free text query string describing the user's desired characteristics.\n\n"
    "Your task:\n"
    "Use the article objects as the primary evidence to produce a concise AI overview summary that does TWO things:\n"
    "(A) Assesses the health of the user's current portfolio based on the portfolio ticker articles, "
    "noting overall stability and flagging any specific holdings that appear to need attention "
    "(e.g., negative news, deteriorating fundamentals, regulatory or competitive risks, elevated volatility signals).\n"
    "(B) Explains WHY each output ticker is a good match for the portfolio tickers and/or the expanded free text query string.\n\n"
    "Summary rules:\n"
    "- Use SEMANTIC reasoning, not exact keyword overlap only.\n"
    "- Reference the themes, risks, business characteristics, and signals present in each ticker's articles.\n"
    "- For the portfolio assessment: give an overall stability read (e.g., stable, mixed, fragile) and call out by ticker any holdings that warrant attention, with a brief evidence-based reason. If no holdings warrant attention, say so explicitly.\n"
    "- For the output tickers: explicitly connect each output ticker to the relevant portfolio ticker(s) and/or to the free text query.\n"
    "- Be conservative, factual, and grounded in the article evidence. Do not invent facts or give buy/sell recommendations.\n"
    "- Keep the tone neutral and analytical.\n"
    "- Cover every output ticker at least once.\n"
    "- Do NOT add any new tickers.\n"
    "- Do NOT omit any output tickers.\n"
    "- Do NOT return scores, rankings, bullet lists, JSON, or markdown formatting.\n"
    "- Do NOT include disclaimers, preambles, or closing remarks (e.g., 'Here is a summary...').\n"
    "- Return ONLY the summary as a single plain text string.\n\n"
    "Structure and length:\n"
    "- Write as flowing prose in 2 short paragraphs separated by a single blank line.\n"
    "- Paragraph 1 (2-4 sentences): portfolio health assessment and any tickers needing attention.\n"
    "- Paragraph 2 (3-5 sentences): alignment rationale for the output tickers.\n\n"
    "Respond ONLY with the plain text summary string. Example shape (illustrative only):\n"
    '"The portfolio reads as broadly stable, with consistent coverage of durable enterprise software and '
    "semiconductor names and no widespread negative signals. That said, NVDA warrants attention given "
    "recurring articles on export-control exposure and customer concentration, and TSLA shows mixed signals "
    "tied to margin pressure and demand softness in recent coverage.\\n\\n"
    "PANW aligns with the portfolio's cybersecurity exposure given recurring articles on enterprise threat "
    "detection and platform consolidation, themes that also appear in the query's emphasis on defensive "
    "software. QCOM matches the portfolio's semiconductor and mobile connectivity tilt, with article coverage "
    "highlighting handset recovery and edge-AI design wins. ACN reinforces the query's focus on enterprise "
    "digital transformation, echoing consulting-led AI deployment themes seen across portfolio holdings. "
    'CSCO complements the networking and infrastructure signals present in both the portfolio articles and the query."\n'
)


def get_ai_overview(
    portfolio_tickers=[],
    output_tickers=[],
    client=None,
    free_text_query="",
):
    portfolio_ticker_docs = _build_ticker_docs(
        portfolio_tickers, max_articles_per_ticker=20, max_summary_chars=150, max_headline_chars=100, summary_key="text"
    )

    output_ticker_docs = _build_ticker_docs(
        output_tickers, max_articles_per_ticker=20, max_summary_chars=150, max_headline_chars=100, summary_key="text"
    )

    response = client.chat(
        [
            {"role": "system", "content": _AI_OVERVIEW_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Portfolio ticker articles:\n{json.dumps(portfolio_ticker_docs)}\n\n"
                    f"Output ticker articles:\n{json.dumps(output_ticker_docs)}\n"
                    f"Expanded free text query:\n{expand_stock_query(free_text_query, client)}\n\n"
                ),
            },
        ]
    )
    parsed = re.sub(r"\s+", " ", (response.get("content") or "").strip())

    return parsed


def expand_stock_query(user_query, client):
    key = (user_query or "").strip().lower()
    if not key:
        return ""

    if key in _EXPAND_CACHE:
        return _EXPAND_CACHE[key]

    response = client.chat(
        [
            {"role": "system", "content": _EXPAND_STOCK_QUERY_PROMPT},
            {"role": "user", "content": f"User stock preference query: {user_query}"},
        ]
    )
    parsed = re.sub(r"\s+", " ", (response.get("content") or "").strip())

    if len(_EXPAND_CACHE) >= _EXPAND_CACHE_MAX:
        _EXPAND_CACHE.pop(next(iter(_EXPAND_CACHE)), None)
    _EXPAND_CACHE[key] = parsed

    return parsed


def get_risk_signals_for_tickers(tickers, client):
    ticker_docs = _build_ticker_docs(
        tickers, max_articles_per_ticker=30, max_summary_chars=200, max_headline_chars=150, summary_key="text"
    )

    response = client.chat(
        [
            {"role": "system", "content": _RISK_SIGNALS_TICKERS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Risk word bank:\n{RISK_WORD_DICT_JSON}\n\n" f"Ticker articles:\n{json.dumps(ticker_docs)}\n"
                ),
            },
        ]
    )
    parsed = _parse_json_response((response.get("content") or "").strip(), expected="object")

    return {
        "signals": parsed if isinstance(parsed, dict) else {},
        "ticker_docs": ticker_docs,
    }


def get_ticker_summary(tickers, client, positive_bias=False):
    ticker_docs = _build_ticker_docs(
        tickers, max_articles_per_ticker=30, max_summary_chars=180, max_headline_chars=120, summary_key="text"
    )

    response = client.chat(
        [
            {"role": "system", "content": _TICKERS_SUMMARY_PROMPT},
            {
                "role": "user",
                "content": (f"Ticker articles:\n{json.dumps(ticker_docs)}\n" f"Positive bias: {positive_bias}\n"),
            },
        ]
    )
    parsed = _parse_json_response((response.get("content") or "").strip(), expected="object")
    return parsed if isinstance(parsed, dict) else {}


def get_ai_ticker_ranking(
    tickers,
    client,
    free_text_query="",
    max_articles_per_ticker=5,
    max_summary_chars=200,
    max_headline_chars=150,
):
    ticker_docs = _build_ticker_docs(
        tickers,
        max_articles_per_ticker=max_articles_per_ticker,
        max_summary_chars=max_summary_chars,
        max_headline_chars=max_headline_chars,
        summary_key="summary",
    )

    response = client.chat(
        [
            {"role": "system", "content": _AI_TICKER_RANKING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Expanded free text query:\n{expand_stock_query(free_text_query, client)}\n\n"
                    f"Tickers:\n{json.dumps(tickers)}\n\n"
                    f"Ticker articles:\n{json.dumps(ticker_docs)}\n"
                ),
            },
        ]
    )
    parsed = _parse_json_response(response.get("content"), expected="array")
    return parsed if isinstance(parsed, list) else []


def _get_client():
    return LLMClient(api_key=os.getenv("SPARK_API_KEY"))
