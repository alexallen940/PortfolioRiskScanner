import csv
import re
import difflib
from urllib.parse import urlparse
from collections import Counter
from infosci_spark_client import LLMClient
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm_routes import (
    get_ai_ticker_ranking,
    get_risk_signals_for_tickers,
    get_ticker_summary,
    expand_stock_query,
)
from services.svd import get_fitted_svd
from models import Article, RiskData
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import os
import traceback


try:
    import yfinance as yfin
except ImportError:
    yfin = None

# load the risk word bank
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
json_path = os.path.join(BASE_DIR, "data", "risk_word_bank.json")
articles_csv_path = os.path.join(BASE_DIR, "data", "articles.csv")
ticker_metadata_csv_path = os.path.join(BASE_DIR, "data", "ticker_metadata.csv")
constituents_csv_path = os.path.join(BASE_DIR, "data", "constituents.csv")

with open(json_path, "r") as f:
    RISK_KEYWORDS = json.load(f)


GENERIC_NAME_PREFIXES = (
    "A Look At",
    "Assessing",
    "Why",
    "Here",
    "Is",
    "How",
    "What",
    "Tracking",
    "Final Trade",
    "Stock Market Today",
    "Live On",
)


def _clean_company_name(name):
    cleaned = re.sub(r"\s+", " ", name).strip(" ,.-:")

    if not cleaned:
        return None

    if cleaned.startswith(GENERIC_NAME_PREFIXES):
        return None

    if len(cleaned) > 70:
        return None

    return cleaned


def _extract_company_name(text, ticker):
    if not text:
        return None

    patterns = [
        rf"^([A-Z0-9][A-Za-z0-9&.,'\- ]+?)\s+\((?:NYSE|NASDAQ|NasdaqGS|NasdaqGM|NasdaqCM|NYSEARCA):{ticker}\)",
        rf"^([A-Z0-9][A-Za-z0-9&.,'\- ]+?)\s+\({ticker}\)",
        r"^([A-Z0-9][A-Za-z0-9&.,'\- ]+?):",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            cleaned = _clean_company_name(match.group(1))
            if cleaned:
                return cleaned

    return None


def _load_company_metadata_from_dataset():
    metadata = {}

    dataset_path = None
    if os.path.exists(ticker_metadata_csv_path):
        dataset_path = ticker_metadata_csv_path
    elif os.path.exists(constituents_csv_path):
        dataset_path = constituents_csv_path

    if not dataset_path:
        return metadata

    with open(dataset_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = (row.get("ticker") or row.get("Symbol") or row.get("symbol") or "").strip().upper()
            if not ticker:
                continue

            company_name = (
                row.get("company_name") or row.get("Security") or row.get("security") or ""
            ).strip() or ticker
            logo_url = (row.get("logo_url") or "").strip() or None

            # Optional: logo_domain supports Apistemic logos API generation.
            logo_domain = (row.get("logo_domain") or "").strip().lower()
            if logo_domain:
                logo_domain = re.sub(r"^https?://", "", logo_domain)
                logo_domain = logo_domain.split("/")[0]

            if not logo_url and logo_domain:
                logo_url = f"https://logos-api.apistemic.com/domain:{logo_domain}"

            metadata[ticker] = {"company_name": company_name, "logo_url": logo_url}

    return metadata


def _load_company_metadata_from_articles():
    metadata = {}

    with open(articles_csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = row.get("ticker", "").strip().upper()
            if not ticker:
                continue

            entry = metadata.setdefault(
                ticker,
                {"company_name": ticker, "logo_url": None},
            )

            if entry["company_name"] == ticker:
                extracted_name = _extract_company_name(row.get("headline", ""), ticker) or _extract_company_name(
                    row.get("summary", ""), ticker
                )
                if extracted_name:
                    entry["company_name"] = extracted_name

    return metadata


def _load_article_link_lookup():
    article_links = {}

    if not os.path.exists(articles_csv_path):
        return article_links

    with open(articles_csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            headline = (row.get("headline") or "").strip()
            url = (row.get("url") or "").strip() or None

            if ticker and headline and url:
                article_links[(ticker, headline)] = url

    return article_links


def _load_company_metadata():
    dataset_metadata = _load_company_metadata_from_dataset()

    # Use dataset values as source of truth; only fill missing names from article text.
    article_metadata = _load_company_metadata_from_articles()
    combined = {ticker: values.copy() for ticker, values in dataset_metadata.items()}

    for ticker, article_values in article_metadata.items():
        entry = combined.setdefault(ticker, {"company_name": ticker, "logo_url": None})
        if entry.get("company_name", ticker) == ticker and article_values.get("company_name"):
            entry["company_name"] = article_values["company_name"]

    return combined


COMPANY_METADATA = _load_company_metadata()
YFINANCE_METADATA_CACHE = {}
ARTICLE_LINK_LOOKUP = _load_article_link_lookup()


def _fuzzy_correct_query_text(query_text, unigram_features, min_length=4, cutoff=0.84):
    if not query_text:
        return "", {}

    feature_set = set(unigram_features)
    corrections = {}

    def _replace(match):
        token = match.group(0)
        lower = token.lower()

        if len(lower) < min_length or lower in feature_set:
            return token

        close_match = difflib.get_close_matches(lower, unigram_features, n=1, cutoff=cutoff)
        if not close_match:
            return token

        corrected = close_match[0]
        if corrected == lower:
            return token

        corrections[token] = corrected
        return corrected

    corrected_query = re.sub(r"\b[A-Za-z][A-Za-z']+\b", _replace, query_text)
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
        parsed = urlparse(candidate)
        domain = (parsed.netloc or "").lower()
    except Exception:
        return None

    if domain.startswith("www."):
        domain = domain[4:]

    # Strip non-brand subdomains (e.g. corporate.lululemon.com → lululemon.com)
    # Keep only the registrable domain (last two labels: example.com)
    parts = domain.split(".")
    if len(parts) > 2:
        domain = ".".join(parts[-2:])

    return domain or None


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


SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()


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

    article_sentiments = []
    for article in articles:
        text = f"{article.headline}. {article.summary}"
        sentiment = get_article_sentiment(text)

        article_sentiments.append(
            {
                "headline": article.headline,
                "sentiment": sentiment,
            }
        )

    avg_compound = sum(item["sentiment"]["compound"] for item in article_sentiments) / len(article_sentiments)

    if avg_compound >= 0.5:
        overall_label = "very positive"
    elif avg_compound >= 0.15:
        overall_label = "positive"
    elif avg_compound >= 0.05:
        overall_label = "slightly positive"
    elif avg_compound <= -0.5:
        overall_label = "very negative"
    elif avg_compound <= -0.15:
        overall_label = "negative"
    elif avg_compound <= -0.05:
        overall_label = "slightly negative"
    else:
        overall_label = "neutral"

    return {
        "ticker": normalized_ticker,
        "article_count": len(article_sentiments),
        "average_compound": round(avg_compound, 4),
        "label": overall_label,
        "articles": article_sentiments,
    }


def _build_risk_breakdown(risk_row, min_raw_score, max_raw_score):
    if not risk_row:
        return None

    annualized_volatility = float(risk_row.annualized_volatility)
    max_drawdown_abs = abs(float(risk_row.max_drawdown))
    var_95_abs = abs(float(risk_row.var_95))
    downside_volatility = float(risk_row.downside_volatility)
    avg_daily_volume = float(risk_row.avg_daily_volume)

    weighted_volatility = 0.30 * annualized_volatility
    weighted_drawdown = 0.25 * max_drawdown_abs
    weighted_var_95 = 0.20 * var_95_abs
    weighted_downside = 0.15 * downside_volatility
    weighted_volume_inverse = 0.10 * (1 / (avg_daily_volume + 1))

    raw_score_formula = (
        weighted_volatility + weighted_drawdown + weighted_var_95 + weighted_downside + weighted_volume_inverse
    )

    denominator = max_raw_score - min_raw_score
    if denominator > 0:
        normalized_from_formula = 1 + 9 * ((raw_score_formula - min_raw_score) / denominator)
    else:
        normalized_from_formula = float(risk_row.risk_score_1_10)

    return {
        "weights": {
            "annualized_volatility": 0.30,
            "max_drawdown": 0.25,
            "var_95": 0.20,
            "downside_volatility": 0.15,
            "avg_daily_volume_inverse": 0.10,
        },
        "components": {
            "annualized_volatility": annualized_volatility,
            "max_drawdown_abs": max_drawdown_abs,
            "var_95_abs": var_95_abs,
            "downside_volatility": downside_volatility,
            "avg_daily_volume": avg_daily_volume,
            "avg_daily_volume_inverse": 1 / (avg_daily_volume + 1),
        },
        "weighted_components": {
            "annualized_volatility": weighted_volatility,
            "max_drawdown": weighted_drawdown,
            "var_95": weighted_var_95,
            "downside_volatility": weighted_downside,
            "avg_daily_volume_inverse": weighted_volume_inverse,
        },
        "raw_score": float(risk_row.raw_risk_score),
        "raw_score_from_formula": raw_score_formula,
        "min_raw_score": min_raw_score,
        "max_raw_score": max_raw_score,
        "normalized_score": float(risk_row.risk_score_1_10),
        "normalized_score_from_formula": normalized_from_formula,
    }


def _build_similarity_explanation(
    idx,
    score,
    use_svd,
    query_repr,
    doc_repr,
    vectorizer,
    portfolio_weight,
    text_weight,
    text_weight_level,
    svd=None,
):
    if use_svd and svd is not None:
        q = query_repr.flatten()
        d = doc_repr[idx].flatten()
        products = q * d
        dot_product = float(np.dot(q, d))
        query_norm = float(np.linalg.norm(q))
        stock_norm = float(np.linalg.norm(d))
        denominator = query_norm * stock_norm
        top_dims = np.argsort(np.abs(products))[::-1][:5]
        feature_names = vectorizer.get_feature_names_out()

        top_drivers = []
        for dim in top_dims:
            if q[dim] > 0 and d[dim] > 0:
                relationship = "both positive"
            elif q[dim] < 0 and d[dim] < 0:
                relationship = "both negative"
            else:
                relationship = "opposite signs"

            component = svd.components_[dim]
            top_pos_idx = component.argsort()[-4:][::-1]
            top_neg_idx = component.argsort()[:4]
            pos_terms = [feature_names[j].replace("_", " ") for j in top_pos_idx]
            neg_terms = [feature_names[j].replace("_", " ") for j in top_neg_idx]
            label_terms = pos_terms[:3]
            if label_terms:
                label = "Theme: " + ", ".join(label_terms)
            else:
                label = "Theme: mixed market language"

            top_drivers.append(
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

        return {
            "method": "svd_cosine",
            "similarity_score": float(score),
            "dot_product": dot_product,
            "query_norm": query_norm,
            "stock_norm": stock_norm,
            "denominator": denominator,
            "portfolio_weight": portfolio_weight,
            "text_weight": text_weight,
            "text_weight_level": text_weight_level,
            "top_drivers": top_drivers,
        }

    query_vec = query_repr[0]
    stock_vec = doc_repr[idx]
    overlap = query_vec.multiply(stock_vec).tocoo()
    dot_product = float(query_vec.multiply(stock_vec).sum())
    query_norm = float(np.sqrt(query_vec.multiply(query_vec).sum()))
    stock_norm = float(np.sqrt(stock_vec.multiply(stock_vec).sum()))
    denominator = query_norm * stock_norm
    feature_names = vectorizer.get_feature_names_out()

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
        "denominator": denominator,
        "portfolio_weight": portfolio_weight,
        "text_weight": text_weight,
        "text_weight_level": text_weight_level,
        "top_drivers": top_drivers,
    }


def get_stock_recommendations(
    user_portfolio,
    desired_characteristics="",
    top_k=4,
    vectorizer=TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3), min_df=10),
    use_svd=True,
    n_components=20,
    portfolio_weight=1,
    text_weight=150,
    text_weight_level="medium",
    use_llm=True,
):
    ticker_docs = {}
    articles = Article.query.all()

    for article in articles:
        # combine headline and summary
        text = f"{article.headline} {article.summary}"

        if article.ticker not in ticker_docs:
            ticker_docs[article.ticker] = []

        # add the article text under that ticker
        ticker_docs[article.ticker].append(text)

    # convert lists of article texts into one big doc per ticker
    ticker_docs = {
        ticker: " ".join(texts).replace('"', "").replace("-", " ").lower().strip()
        for ticker, texts in ticker_docs.items()
    }

    tickers = list(ticker_docs.keys())
    documents = list(ticker_docs.values())
    risk_rows = RiskData.query.all()
    risk_by_ticker = {row.ticker: row for row in risk_rows}
    risk_scores = {row.ticker: row.risk_score_1_10 for row in risk_rows}
    min_raw_score = min((float(row.raw_risk_score) for row in risk_rows), default=0.0)
    max_raw_score = max((float(row.raw_risk_score) for row in risk_rows), default=0.0)

    # has 503 rows (one entry for each stock) and a certain number of words or phrases for columns
    tfidf_matrix = vectorizer.fit_transform(documents)

    portfolio_texts = []
    for ticker in user_portfolio:
        if ticker in ticker_docs:
            portfolio_texts.append(ticker_docs[ticker])

    unigram_features = [feature for feature in vectorizer.get_feature_names_out() if "_" not in feature]
    query_for_retrieval = desired_characteristics

    if use_llm:
        api_key = os.getenv("SPARK_API_KEY")
        try:
            client = LLMClient(api_key=api_key)
            # query expansion step
            query_for_retrieval = expand_stock_query(desired_characteristics, client)
        except Exception:
            query_for_retrieval = desired_characteristics

    corrected_characteristics, query_corrections = _fuzzy_correct_query_text(query_for_retrieval, unigram_features)

    # free text query
    characteristics_doc = corrected_characteristics.replace('"', "").replace("-", " ").lower().strip()

    # case when no valid portfolio tickers or free text query were found
    if not portfolio_texts and not characteristics_doc:
        return []

    # combine all portfolio ticker docs into one big doc
    portfolio_doc = " ".join(portfolio_texts).replace('"', "").replace("-", " ").lower().strip()

    # weighted combined query
    combined_query_doc = (
        ((portfolio_doc + " ") * portfolio_weight) + ((characteristics_doc + " ") * text_weight)
    ).strip()

    # same TF-IDF space as the S&P 500
    portfolio_vector = vectorizer.transform([combined_query_doc])

    if use_svd:
        svd = get_fitted_svd(ticker_docs, combined_query_doc, vectorizer, n_components)
        doc_repr = svd.transform(tfidf_matrix)
        query_repr = svd.transform(portfolio_vector)

        # length 503 (for each stock in the S&P 500's similarity score to portfolio)
        similarities = cosine_similarity(query_repr, doc_repr).flatten()

        # print("\n========== SVD SEARCH RESULTS ==========")

        # print top results
        sorted_tickers_idx = np.argsort(similarities)[::-1]

        recommendations_ind = []
        for idx in sorted_tickers_idx:
            if tickers[idx] not in user_portfolio:
                recommendations_ind.append(idx)

        recommendations_ind = recommendations_ind[:4]

        # print("\nTop Stock Recommendations:")
        # for idx in recommendations_ind:
        #     print(f"{tickers[idx]} -> similarity {similarities[idx]:.4f}")

        q = query_repr.flatten()

        for idx in recommendations_ind:
            d = doc_repr[idx].flatten()

            explain_recommendation(idx, q, d, tickers, vectorizer, svd, top_k_dims=5)

    else:
        doc_repr = tfidf_matrix
        query_repr = portfolio_vector

        # length 503 (for each stock in the S&P 500's similarity score to portfolio)
        similarities = cosine_similarity(query_repr, doc_repr).flatten()

    results = []
    for idx, score in enumerate(similarities):
        ticker = tickers[idx]
        if ticker not in user_portfolio:
            company_metadata = COMPANY_METADATA.get(ticker, {})
            risk_row = risk_by_ticker.get(ticker)
            results.append(
                {
                    "ticker": ticker,
                    "similarity": float(score),
                    "similarity_explanation": _build_similarity_explanation(
                        idx,
                        score,
                        use_svd,
                        query_repr,
                        doc_repr,
                        vectorizer,
                        portfolio_weight,
                        text_weight,
                        text_weight_level,
                        svd=svd if use_svd else None,
                    ),
                    "risk_score": risk_scores.get(ticker),
                    "risk_breakdown": _build_risk_breakdown(risk_row, min_raw_score, max_raw_score),
                    "company_name": company_metadata.get("company_name", ticker),
                    "logo_url": company_metadata.get("logo_url"),
                }
            )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = results[:top_k]
    for item in top_results:
        sentiment_summary = get_ticker_sentiment_summary(item["ticker"], max_articles=10)
        item["sentiment"] = {
            "label": sentiment_summary["label"],
            "average_compound": sentiment_summary["average_compound"],
            "article_count": sentiment_summary["article_count"],
        }
    top_results = _enrich_with_yfinance(top_results)

    # Preserve IR-only order before any LLM reranking
    ir_results = list(top_results)

    if use_llm:
        try:
            api_key = os.getenv("SPARK_API_KEY")
            client = LLMClient(api_key=api_key)

            # print("IR tickers:", [result["ticker"] for result in top_results])
            ranking = get_ai_ticker_ranking([result["ticker"] for result in top_results], client, query_for_retrieval)

            temp = []
            for ticker in ranking:
                for result in top_results:
                    if result["ticker"] == ticker:
                        temp.append(result)

            top_results = temp if len(temp) == len(top_results) else top_results

        except Exception:
            traceback.print_exc()

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


def explain_recommendation(idx, q=[], d=[], tickers=[], vectorizer=None, svd=None, top_k_dims=3):

    # print(f"\n--- {tickers[idx]} ---")

    products = q * d
    top_dims = np.argsort(np.abs(products))[::-1][:top_k_dims]

    feature_names = vectorizer.get_feature_names_out()

    for dim in top_dims:
        relation = (
            "both positive"
            if (q[dim] > 0 and d[dim] > 0)
            else ("both negative" if (q[dim] < 0 and d[dim] < 0) else "opposite signs")
        )

        component = svd.components_[dim]

        top_pos_idx = component.argsort()[-5:][::-1]
        top_neg_idx = component.argsort()[:5]

        pos_terms = [feature_names[j] for j in top_pos_idx]
        neg_terms = [feature_names[j] for j in top_neg_idx]

        # print(f"\nDimension {dim}")
        # print(f"  Query value: {q[dim]:.4f}")
        # print(f"  Stock value: {d[dim]:.4f}")
        # print(f"  Product: {products[dim]:.4f}")
        # print(f"  Relationship: {relation}")
        # print(f"  Positive terms: {pos_terms}")
        # print(f"  Negative terms: {neg_terms}")


def get_recommendation_desc(ticker, max_articles=25, use_llm=True, top_k_risk_signals=2, top_k_risk_types=2):
    bullets = []
    details = []
    normalized_ticker = ticker.upper().strip()

    if use_llm:
        try:
            api_key = os.getenv("SPARK_API_KEY")
            client = LLMClient(api_key=api_key)

            get_ticker_summary(tickers=[normalized_ticker], client=client)

            data = get_risk_signals_for_tickers(tickers=[normalized_ticker], client=client)

            signals_data = data.get("signals")
            ticker_docs = data.get("ticker_docs")

            for ticker, risk_types in signals_data.items():
                docs = ticker_docs.get(ticker)
                top_risks = sorted(
                    risk_types.items(),
                    key=lambda kv: sum(signal["count"] for signal in kv[1].values()),
                    reverse=True,
                )[:top_k_risk_types]

                for risk_type_idx, (risk_type, signals) in enumerate(top_risks):
                    if not signals:
                        continue
                    top_signal_items = sorted(
                        signals.items(),
                        key=lambda kv: kv[1]["count"],
                        reverse=True,
                    )[:top_k_risk_signals]

                    signal_names = [signal for signal, _ in top_signal_items]
                    signal_text = " and ".join(signal_names)

                    article_indices = []
                    for _, info in top_signal_items:
                        for i in info.get("article_indices"):
                            if i not in article_indices:
                                article_indices.append(i)

                    headlines = [
                        {
                            "title": docs[i]["headline"],
                            "url": ARTICLE_LINK_LOOKUP.get((ticker, docs[i]["headline"])),
                        }
                        for i in article_indices
                        if 0 <= i < len(docs) and docs[i].get("headline")
                    ][:3]

                    if risk_type_idx == 0:
                        bullet = f"{risk_type} due to {signal_text} risk signals in recent news coverage"
                    else:
                        bullet = f"{risk_type} due to {signal_text} risk signals"

                    bullets.append(bullet)
                    details.append(
                        {
                            "bullet": bullet,
                            "headlines": headlines,
                        }
                    )

        except Exception:
            traceback.print_exc()

    if not use_llm or (not details and not bullets):
        # take 'max_articles' number of most recent articles for ticker
        articles = (
            Article.query.filter_by(ticker=normalized_ticker).order_by(Article.id.desc()).limit(max_articles).all()
        )

        # counts number of times each risk signal appears
        risk_counts = Counter()

        # keeps track of which keyword within the risk signal was matched
        keyword_hits = {}

        # tracks which article headlines matched each risk type (up to 3 per type)
        headline_hits = {}

        # count keyword matches per article
        for article in articles:
            article_text = f"{article.headline} {article.summary}".replace("-", " ").lower()
            matched_types = set()
            for risk_type, keywords in RISK_KEYWORDS.items():
                for keyword in keywords:
                    if re.search(rf"\b{re.escape(keyword)}\b", article_text):
                        risk_counts[risk_type] += 1
                        if risk_type not in keyword_hits:
                            keyword_hits[risk_type] = []
                        keyword_hits[risk_type].append(keyword)
                        matched_types.add(risk_type)

            for risk_type in matched_types:
                if risk_type not in headline_hits:
                    headline_hits[risk_type] = []
                if len(headline_hits[risk_type]) < 3:
                    headline_hits[risk_type].append(
                        {
                            "title": article.headline,
                            "url": ARTICLE_LINK_LOOKUP.get((normalized_ticker, article.headline)),
                        }
                    )

        # the case where no keywords match
        if not risk_counts:
            no_signal_result = {
                "bullet": "no apparent risk themes based on recent news. need more info to generate summary.",
                "headlines": [],
            }
            return {
                "bullets": [no_signal_result["bullet"]],
                "details": [no_signal_result],
            }

        # top 2 risk types
        sort_signals = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)
        top_risks = [r for r, _ in sort_signals[:top_k_risk_types]]

        for i, risk in enumerate(top_risks):
            keywords = list(dict.fromkeys(keyword_hits.get(risk, [])))[:top_k_risk_signals]
            if keywords:
                keyword_text = " and ".join(keywords)
                if i == 0:
                    bullet = f"{risk} due to {keyword_text} risk signals in recent news coverage"
                else:
                    bullet = f"{risk} due to {keyword_text} risk signals"
            else:
                if i == 0:
                    bullet = f"Susceptible to {risk}"
                else:
                    bullet = f"Susceptible to {risk}"
            bullets.append(bullet)
            details.append(
                {
                    "bullet": bullet,
                    "headlines": headline_hits.get(risk, []),
                }
            )

    return {"bullets": bullets, "details": details}
