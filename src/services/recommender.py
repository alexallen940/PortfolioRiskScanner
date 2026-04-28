import difflib
import os
import re
import traceback
from collections import Counter, defaultdict
from urllib.parse import urlparse
import numpy as np
from infosci_spark_client import LLMClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from models import Article
from services.llm_services import (
    expand_stock_query,
    get_ai_ticker_ranking,
    get_risk_signals_for_tickers,
    get_ticker_summary,
)
from utils.load_from_db import RISK_KEYWORDS
from utils.recommendation_index import RecommendationIndex


try:
    import yfinance as yfin
except ImportError:
    yfin = None


INDEX = RecommendationIndex()
YFINANCE_METADATA_CACHE = {}

SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()

TICKER_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z']+\b")


def _fuzzy_correct_query_text(query_text, unigram_features=None, min_length=4, cutoff=0.84):
    if not query_text:
        return "", {}

    if unigram_features is None:
        unigram_features = INDEX.unigram_features

    feature_set = set(unigram_features)
    corrections = {}

    def _replace(match):
        token = match.group(0)
        lower = token.lower()

        if len(lower) < min_length or lower in feature_set:
            return token

        close = difflib.get_close_matches(lower, unigram_features, n=1, cutoff=cutoff)
        if not close or close[0] == lower:
            return token

        corrections[token] = close[0]
        return close[0]

    corrected_query = TICKER_WORD_RE.sub(_replace, query_text)
    return corrected_query, corrections


def _website_to_domain(website):
    if not website:
        return None

    candidate = website.strip()
    if not candidate:
        return None

    if not re.match(r"^https?://", candidate):
        candidate = f"https://{candidate}"

    try:
        domain = (urlparse(candidate).netloc or "").lower()
    except Exception:
        return None

    if domain.startswith("www."):
        domain = domain[4:]

    # Keep only the registrable domain (last two labels: example.com)
    parts = domain.split(".")
    if len(parts) > 2:
        domain = ".".join(parts[-2:])

    return domain or None


def get_article_url(ticker, headline):
    article_link_lookup = INDEX.article_link_lookup
    exact_url = article_link_lookup.get((ticker, headline))
    if exact_url:
        return exact_url

    normalized_headline = headline.strip().lower()
    for (stored_ticker, stored_headline), url in article_link_lookup.items():
        if stored_ticker == ticker and stored_headline.strip().lower() == normalized_headline:
            return url

    return None


def _fetch_yfinance_metadata(ticker):
    ticker = ticker.upper().strip()
    if not ticker:
        return None

    if ticker in YFINANCE_METADATA_CACHE:
        return YFINANCE_METADATA_CACHE[ticker]

    if yfin is None:
        YFINANCE_METADATA_CACHE[ticker] = None
        return None

    try:
        info = yfin.Ticker(ticker).info or {}
        long_name = (info.get("longName") or info.get("shortName") or "").strip() or None

        logo_url = (info.get("logo_url") or "").strip() or None
        if not logo_url:
            domain = _website_to_domain(info.get("website"))
            if domain:
                logo_url = f"https://logos-api.apistemic.com/domain:{domain}"

        metadata = {"company_name": long_name, "logo_url": logo_url}
        YFINANCE_METADATA_CACHE[ticker] = metadata
        return metadata
    except Exception:
        YFINANCE_METADATA_CACHE[ticker] = None
        return None


def _enrich_with_yfinance(results):
    enriched = []
    for item in results:
        yf_data = _fetch_yfinance_metadata(item.get("ticker", ""))
        if not yf_data:
            enriched.append(item)
            continue

        merged = item.copy()

        # Prefer yfinance for company name to avoid incorrect article-derived names.
        if yf_data.get("company_name"):
            merged["company_name"] = yf_data["company_name"]

        # Keep existing logo_url if explicitly provided; otherwise use yfinance/domain-derived logo.
        if not merged.get("logo_url") and yf_data.get("logo_url"):
            merged["logo_url"] = yf_data["logo_url"]

        enriched.append(merged)

    return enriched


def get_signal_count(signal):
    if isinstance(signal, dict):
        return signal.get("count", 0)
    if isinstance(signal, int):
        return signal
    return 0


def get_article_sentiment(text):
    scores = SENTIMENT_ANALYZER.polarity_scores(text or "")
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "compound": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
    }


def _overall_sentiment_label(avg_compound):
    if avg_compound >= 0.5:
        return "very positive"
    if avg_compound >= 0.15:
        return "positive"
    if avg_compound >= 0.05:
        return "slightly positive"
    if avg_compound <= -0.5:
        return "very negative"
    if avg_compound <= -0.15:
        return "negative"
    if avg_compound <= -0.05:
        return "slightly negative"
    return "neutral"


def get_ticker_sentiment_summary(ticker, max_articles=25):
    normalized_ticker = ticker.upper().strip()

    articles = Article.query.filter_by(ticker=normalized_ticker).order_by(Article.id.desc()).limit(max_articles).all()

    if not articles:
        return {
            "ticker": normalized_ticker,
            "article_count": 0,
            "average_compound": 0.0,
            "label": "neutral",
            "articles": [],
        }

    article_sentiments = [
        {
            "headline": article.headline,
            "sentiment": get_article_sentiment(f"{article.headline}. {article.summary}"),
        }
        for article in articles
    ]

    avg_compound = sum(item["sentiment"]["compound"] for item in article_sentiments) / len(article_sentiments)

    return {
        "ticker": normalized_ticker,
        "article_count": len(article_sentiments),
        "average_compound": round(avg_compound, 4),
        "label": _overall_sentiment_label(avg_compound),
        "articles": article_sentiments,
    }


_RISK_BREAKDOWN_WEIGHTS = {
    "annualized_volatility": 0.30,
    "max_drawdown": 0.25,
    "var_95": 0.20,
    "downside_volatility": 0.15,
    "avg_daily_volume_inverse": 0.10,
}


def _build_risk_breakdown(risk_row, min_raw_score, max_raw_score):
    if not risk_row:
        return None

    annualized_volatility = float(risk_row.annualized_volatility)
    max_drawdown_abs = abs(float(risk_row.max_drawdown))
    var_95_abs = abs(float(risk_row.var_95))
    downside_volatility = float(risk_row.downside_volatility)
    avg_daily_volume = float(risk_row.avg_daily_volume)
    avg_daily_volume_inverse = 1 / (avg_daily_volume + 1)

    weighted = {
        "annualized_volatility": _RISK_BREAKDOWN_WEIGHTS["annualized_volatility"] * annualized_volatility,
        "max_drawdown": _RISK_BREAKDOWN_WEIGHTS["max_drawdown"] * max_drawdown_abs,
        "var_95": _RISK_BREAKDOWN_WEIGHTS["var_95"] * var_95_abs,
        "downside_volatility": _RISK_BREAKDOWN_WEIGHTS["downside_volatility"] * downside_volatility,
        "avg_daily_volume_inverse": _RISK_BREAKDOWN_WEIGHTS["avg_daily_volume_inverse"] * avg_daily_volume_inverse,
    }
    raw_score_formula = sum(weighted.values())

    denominator = max_raw_score - min_raw_score
    if denominator > 0:
        normalized_from_formula = 1 + 9 * ((raw_score_formula - min_raw_score) / denominator)
    else:
        normalized_from_formula = float(risk_row.risk_score_1_10)

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
        "raw_score": float(risk_row.raw_risk_score),
        "raw_score_from_formula": raw_score_formula,
        "min_raw_score": min_raw_score,
        "max_raw_score": max_raw_score,
        "normalized_score": float(risk_row.risk_score_1_10),
        "normalized_score_from_formula": normalized_from_formula,
    }


def _svd_top_drivers(q, d, products, svd, feature_names, top_k=5, top_terms=4):
    top_dims = np.argsort(np.abs(products))[::-1][:top_k]
    drivers = []

    for dim in top_dims:
        if q[dim] > 0 and d[dim] > 0:
            relationship = "both positive"
        elif q[dim] < 0 and d[dim] < 0:
            relationship = "both negative"
        else:
            relationship = "opposite signs"

        component = svd.components_[dim]
        top_pos_idx = component.argsort()[-top_terms:][::-1]
        top_neg_idx = component.argsort()[:top_terms]
        pos_terms = [feature_names[j].replace("_", " ") for j in top_pos_idx]
        neg_terms = [feature_names[j].replace("_", " ") for j in top_neg_idx]

        label_terms = pos_terms[:3]
        label = "Theme: " + (", ".join(label_terms) if label_terms else "mixed market language")

        drivers.append(
            {
                "dimension": int(dim),
                "label": label,
                "query_value": float(q[dim]),
                "stock_value": float(d[dim]),
                "contribution": float(products[dim]),
                "relationship": relationship,
                "top_positive_terms": pos_terms,
                "top_negative_terms": neg_terms,
            }
        )

    return drivers


def _build_similarity_explanation(
    idx,
    score,
    use_svd,
    query_repr,
    doc_repr,
    portfolio_weight,
    text_weight,
    text_weight_level,
    svd=None,
    feature_names=None,
):
    if use_svd and svd is not None:
        q = query_repr.flatten()
        d = doc_repr[idx].flatten()
        products = q * d
        dot_product = float(np.dot(q, d))
        query_norm = float(np.linalg.norm(q))
        stock_norm = float(np.linalg.norm(d))

        return {
            "method": "svd_cosine",
            "similarity_score": float(score),
            "dot_product": dot_product,
            "query_norm": query_norm,
            "stock_norm": stock_norm,
            "denominator": query_norm * stock_norm,
            "portfolio_weight": portfolio_weight,
            "text_weight": text_weight,
            "text_weight_level": text_weight_level,
            "top_drivers": _svd_top_drivers(q, d, products, svd, feature_names),
        }

    query_vec = query_repr[0]
    stock_vec = doc_repr[idx]
    overlap = query_vec.multiply(stock_vec).tocoo()
    dot_product = float(query_vec.multiply(stock_vec).sum())
    query_norm = float(np.sqrt(query_vec.multiply(query_vec).sum()))
    stock_norm = float(np.sqrt(stock_vec.multiply(stock_vec).sum()))

    if overlap.nnz:
        sorted_indices = np.argsort(np.abs(overlap.data))[::-1][:5]
        top_drivers = [
            {
                "term": feature_names[int(overlap.col[i])].replace("_", " "),
                "contribution": float(overlap.data[i]),
            }
            for i in sorted_indices
        ]
    else:
        top_drivers = []

    return {
        "method": "tfidf_cosine",
        "similarity_score": float(score),
        "dot_product": dot_product,
        "query_norm": query_norm,
        "stock_norm": stock_norm,
        "denominator": query_norm * stock_norm,
        "portfolio_weight": portfolio_weight,
        "text_weight": text_weight,
        "text_weight_level": text_weight_level,
        "top_drivers": top_drivers,
    }


def _expand_query_with_llm(desired_characteristics):
    
    try:
        client = LLMClient(api_key=os.getenv("SPARK_API_KEY"))
        return expand_stock_query(desired_characteristics, client)
    except Exception:
        return desired_characteristics


def _llm_rerank(top_results, query_for_retrieval):
    try:
        client = LLMClient(api_key=os.getenv("SPARK_API_KEY"))
        ranking = get_ai_ticker_ranking([r["ticker"] for r in top_results], client, query_for_retrieval)
        top_tickers_by_result = {r["ticker"]: r for r in top_results}
        reranked = [top_tickers_by_result[t] for t in ranking if t in top_tickers_by_result]
        return reranked if len(reranked) == len(top_results) else top_results
    except Exception:
        traceback.print_exc()
        return top_results


def get_stock_recommendations(
    user_portfolio,
    desired_characteristics="",
    top_k=4,
    use_svd=True,
    portfolio_weight=1,
    text_weight=150,
    text_weight_level="medium",
    use_llm=True,
):
    INDEX.ensure_built()
    vectorizer = INDEX.vectorizer
    tickers = INDEX.tickers
    ticker_docs = INDEX.ticker_docs
    feature_names = INDEX.feature_names
    risk_by_ticker = INDEX.risk_by_ticker
    risk_scores = INDEX.risk_scores
    min_raw_score = INDEX.min_raw_score
    max_raw_score = INDEX.max_raw_score
    company_metadata = INDEX.company_metadata

    portfolio_texts = [ticker_docs[t] for t in user_portfolio if t in ticker_docs]

    query_for_retrieval = _expand_query_with_llm(desired_characteristics) if use_llm else desired_characteristics

    corrected_characteristics, query_corrections = _fuzzy_correct_query_text(
        query_for_retrieval, INDEX.unigram_features
    )

    characteristics_doc = corrected_characteristics.replace('"', "").replace("-", " ").lower().strip()

    if not portfolio_texts and not characteristics_doc:
        return []

    portfolio_doc = " ".join(portfolio_texts).replace('"', "").replace("-", " ").lower().strip()

    combined_query_doc = (
        ((portfolio_doc + " ") * portfolio_weight) + ((characteristics_doc + " ") * text_weight)
    ).strip()

    portfolio_vector = vectorizer.transform([combined_query_doc])

    if use_svd:
        svd = INDEX.svd
        doc_repr = INDEX.doc_repr
        query_repr = svd.transform(portfolio_vector)
    else:
        svd = None
        doc_repr = INDEX.tfidf_matrix
        query_repr = portfolio_vector

    similarities = cosine_similarity(query_repr, doc_repr).flatten()

    sorted_idx = np.argsort(similarities)[::-1]
    top_indices = []
    for idx in sorted_idx:
        if tickers[idx] not in user_portfolio:
            top_indices.append(idx)
            if len(top_indices) == top_k:
                break

    top_results = []
    for idx in top_indices:
        ticker = tickers[idx]
        score = similarities[idx]
        company_metadata = company_metadata.get(ticker, {})
        risk_row = risk_by_ticker.get(ticker)

        top_results.append(
            {
                "ticker": ticker,
                "similarity": float(score),
                "similarity_explanation": _build_similarity_explanation(
                    idx,
                    score,
                    use_svd,
                    query_repr,
                    doc_repr,
                    portfolio_weight,
                    text_weight,
                    text_weight_level,
                    svd=svd,
                    feature_names=feature_names,
                ),
                "risk_score": risk_scores.get(ticker),
                "risk_breakdown": _build_risk_breakdown(risk_row, min_raw_score, max_raw_score),
                "company_name": company_metadata.get("company_name", ticker),
                "logo_url": company_metadata.get("logo_url"),
            }
        )

    for item in top_results:
        sentiment_summary = get_ticker_sentiment_summary(item["ticker"], max_articles=10)
        item["sentiment"] = {
            "label": sentiment_summary["label"],
            "average_compound": sentiment_summary["average_compound"],
            "article_count": sentiment_summary["article_count"],
        }

    top_results = _enrich_with_yfinance(top_results)

    # Preserve IR-only order before any LLM reranking.
    ir_results = list(top_results)

    if use_llm:
        top_results = _llm_rerank(top_results, query_for_retrieval)

        # Generate one ticker summary per recommendation and attach it to
        # both ranking views so the frontend can render AI summaries inline.
        tickers_for_summary = [item["ticker"] for item in top_results]
        try:
            client = LLMClient(api_key=os.getenv("SPARK_API_KEY"))
            summary_by_ticker = get_ticker_summary(
                tickers_for_summary,
                client,
                positive_bias=False,
            )
        except Exception as exc:
            traceback.print_exc()
            error_text = str(exc)
            if "429" in error_text:
                fallback = "AI summary temporarily unavailable due to rate limiting. Please retry in a moment."
            else:
                fallback = "AI summary temporarily unavailable."
            summary_by_ticker = {ticker: fallback for ticker in tickers_for_summary}

        for item in top_results:
            summary = summary_by_ticker.get(item["ticker"])
            if isinstance(summary, str) and summary.strip():
                item["llm_summary"] = summary.strip()

        for item in ir_results:
            summary = summary_by_ticker.get(item["ticker"])
            if isinstance(summary, str) and summary.strip():
                item["llm_summary"] = summary.strip()

    return {
        "recommendations": top_results,
        "ir_recommendations": ir_results,
        "query_interpretation": {
            "original": desired_characteristics,
            "expanded": query_for_retrieval if use_llm else None,
            "interpreted": corrected_characteristics,
            "corrections": query_corrections,
            "used_query_expansion": use_llm,
        },
    }


def _llm_risk_bullets(normalized_ticker, top_k_risk_signals, top_k_risk_types, k_headlines):
   
    bullets = []
    details = []

    try:
        client = LLMClient(api_key=os.getenv("SPARK_API_KEY"))
        data = get_risk_signals_for_tickers(tickers=[normalized_ticker], client=client)
        signals_data = data.get("signals") or {}

        for ticker, risk_types in signals_data.items():
            top_risks = sorted(
                risk_types.items(),
                key=lambda kv: sum(get_signal_count(s) for s in kv[1].values()),
                reverse=True,
            )[:top_k_risk_types]

            articles = INDEX.ticker_article_rows.get(ticker, [])

            for risk_type_idx, (risk_type, signals) in enumerate(top_risks):
                if not signals:
                    continue

                top_signal_items = sorted(
                    signals.items(),
                    key=lambda kv: get_signal_count(kv[1]),
                    reverse=True,
                )[:top_k_risk_signals]

                signal_names = [signal for signal, _ in top_signal_items]
                signal_text = " and ".join(signal_names)

                article_indices = []
                for _, info in top_signal_items:
                    if isinstance(info, dict):
                        for i in info.get("article_indices") or []:
                            if i not in article_indices:
                                article_indices.append(i)

                headlines = [
                    {
                        "title": articles[i].headline,
                        "url": get_article_url(ticker, articles[i].headline),
                    }
                    for i in article_indices
                    if 0 <= i < len(articles)
                ][:k_headlines]

                bullet = f"{risk_type} due to {signal_text} risk signals"

                bullets.append(bullet)
                details.append({"bullet": bullet, "headlines": headlines})

    except Exception:
        traceback.print_exc()
        return [], []

    return bullets, details


def _keyword_risk_bullets(normalized_ticker, max_articles, top_k_risk_signals, top_k_risk_types, k_headlines):
    risk_counts = Counter()
    keyword_hits = defaultdict(list)
    headline_hits = defaultdict(list)

    articles = INDEX.ticker_article_rows.get(normalized_ticker, [])[:max_articles]
    for article in articles:
        article_text = f"{article.headline} {article.summary}".replace("-", " ").lower()
        matched_types = set()

        for risk_type, keywords in RISK_KEYWORDS.items():
            pattern = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)
            matches = pattern.findall(article_text)
            if matches:
                risk_counts[risk_type] += len(matches)
                keyword_hits[risk_type].extend(m.lower() for m in matches)
                matched_types.add(risk_type)

        for risk_type in matched_types:
            if len(headline_hits[risk_type]) < k_headlines:
                headline_hits[risk_type].append(
                    {
                        "title": article.headline,
                        "url": INDEX.article_link_lookup.get((normalized_ticker, article.headline)),
                    }
                )

    if not risk_counts:
        no_signal = {
            "bullet": "no apparent risk themes based on recent news. need more info to generate summary.",
            "headlines": [],
        }
        return [no_signal["bullet"]], [no_signal]

    bullets = []
    details = []
    top_risks = [r for r, _ in risk_counts.most_common(top_k_risk_types)]
    for i, risk in enumerate(top_risks):
        keywords = list(dict.fromkeys(keyword_hits.get(risk, [])))[:top_k_risk_signals]
        if keywords:
            bullet = f"{risk} due to {' and '.join(keywords)} risk signals"
        else:
            bullet = f"Susceptible to {risk}"

        bullets.append(bullet)
        details.append({"bullet": bullet, "headlines": headline_hits.get(risk, [])})

    return bullets, details


def get_recommendation_desc(
    ticker,
    max_articles=25,
    use_llm=True,
    top_k_risk_signals=2,
    top_k_risk_types=2,
    k_headlines=3,
):
    normalized_ticker = ticker.upper().strip()

    bullets, details = ([], [])
    if use_llm:
        bullets, details = _llm_risk_bullets(normalized_ticker, top_k_risk_signals, top_k_risk_types, k_headlines)

    if not use_llm or (not bullets and not details):
        bullets, details = _keyword_risk_bullets(
            normalized_ticker, max_articles, top_k_risk_signals, top_k_risk_types, k_headlines
        )

    return {"bullets": bullets, "details": details}
