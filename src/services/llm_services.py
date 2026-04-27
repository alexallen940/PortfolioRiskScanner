from collections import defaultdict
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

_JSON_WORD_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _ticker_docs_from_index(tickers, max_articles_per_ticker=30, max_summary_chars=200, max_headline_chars=150):
    """Build the {ticker: [{headline, text}, ...]} dict from cached INDEX data."""
    from services.recommender import INDEX

    INDEX.ensure_built()
    res = defaultdict(list)
    for ticker in tickers:
        normalized = ticker.upper().strip()
        docs = INDEX.ticker_article_rows.get(normalized, [])[:max_articles_per_ticker]
        for article in docs:
            res[normalized].append(
                {
                    "headline": (article.headline or "")[:max_headline_chars].strip().lower(),
                    "text": (article.summary or "")[:max_summary_chars].strip().lower(),
                }
            )
    return res


def _parse_json_response(content, expected="object"):
    """Returns parsed value or None"""

    if not content:
        return None

    stripped = _JSON_WORD_RE.sub("", content).strip()

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
    "Do not explain your reasoning.\n"
    "Return ONLY a short expanded query string."
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


def expand_stock_query(user_query, client):
    key = (user_query or "").strip().lower()

    if not key:
        return ""

    if key in _EXPAND_CACHE:
        return _EXPAND_CACHE[key]

    messages = [
        {
            "role": "system",
            "content": _EXPAND_STOCK_QUERY_PROMPT,
        },
        {
            "role": "user",
            "content": f"User stock preference query: {user_query}",
        },
    ]

    response = client.chat(messages)
    content = (response.get("content") or "").strip()
    parsed = re.sub(r"\s+", " ", content)

    if len(_EXPAND_CACHE) >= _EXPAND_CACHE_MAX:
        _EXPAND_CACHE.pop(next(iter(_EXPAND_CACHE)), None)

    _EXPAND_CACHE[key] = parsed

    return parsed


def get_risk_signals_for_tickers(tickers, client):

    ticker_docs = _ticker_docs_from_index(
        tickers, max_articles_per_ticker=40, max_summary_chars=200, max_headline_chars=150
    )

    messages = [
        {"role": "system", "content": _RISK_SIGNALS_TICKERS_PROMPT},
        {
            "role": "user",
            "content": (f"Risk word bank:\n{RISK_WORD_DICT_JSON}\n\n" f"Ticker articles:\n{json.dumps(ticker_docs)}\n"),
        },
    ]

    response = client.chat(messages)
    content = (response.get("content") or "").strip()

    parsed = _parse_json_response(content, expected="object")

    return {
        "signals": parsed if isinstance(parsed, dict) else {},
        "ticker_docs": ticker_docs,
    }


def get_ticker_summary(tickers, client, positive_bias=False):

    ticker_docs = _ticker_docs_from_index(
        tickers, max_articles_per_ticker=12, max_summary_chars=180, max_headline_chars=120
    )

    messages = [
        {"role": "system", "content": _TICKERS_SUMMARY_PROMPT},
        {
            "role": "user",
            "content": (
                f"Ticker articles:\n{json.dumps(ticker_docs)}\n"
                f"Positive bias: {positive_bias}\n"
            ),
        },
    ]

    response = client.chat(messages)
    content = (response.get("content") or "").strip()
    parsed = _parse_json_response(content, expected="object")
    return parsed if isinstance(parsed, dict) else {}


def get_ai_ticker_ranking(
    tickers, client, free_text_query="", max_articles_per_ticker=5, max_summary_chars=200, max_headline_chars=150
):

    from services.recommender import INDEX

    INDEX.ensure_built()

    ticker_docs = defaultdict(list)

    for ticker in tickers:
        normalized = ticker.upper().strip()
        docs = INDEX.ticker_article_rows.get(normalized)
        ticker_docs[ticker] = [
            {
                "headline": (article.headline or "")[:max_headline_chars].strip().lower(),
                "summary": (article.summary or "")[:max_summary_chars].strip().lower(),
            }
            for article in docs[:max_articles_per_ticker]
        ]

    messages = [
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

    response = client.chat(messages)
    content = response.get("content")

    parsed = _parse_json_response(content, expected="array")
    return parsed if isinstance(parsed, list) else []


def _get_client():
    return LLMClient(api_key=os.getenv("SPARK_API_KEY"))
