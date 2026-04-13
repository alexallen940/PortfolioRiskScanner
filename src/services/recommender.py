import csv
import re
from urllib.parse import urlparse
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from services.svd import get_fitted_svd
from models import Article, RiskData
import json
import os

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

json_path = os.path.join(BASE_DIR, "data", "protected_words.json")

with open(json_path, "r") as f:
    PROTECTED_WORDS = json.load(f)


def protect_bigrams(text, protected):
    for bigram in protected:
        text = re.sub(rf"\b{bigram}\b", bigram.replace(" ", "_"), text, flags=re.IGNORECASE)
    return text


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
                (row.get("company_name") or row.get("Security") or row.get("security") or "").strip() or ticker
            )
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
        weighted_volatility
        + weighted_drawdown
        + weighted_var_95
        + weighted_downside
        + weighted_volume_inverse
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
    top_k=10,
    vectorizer=TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3), min_df=10),
    use_svd=True,
    n_components=20,
    portfolio_weight=1,
    text_weight=150,
    text_weight_level="medium",
):
    ticker_docs = {}
    articles = Article.query.all()

    for article in articles:
        # combine headline and summary
        text = f"{article.headline} {article.summary}"

        if article.ticker not in ticker_docs:
            ticker_docs[article.ticker] = []

        # add the article text under that ticker
        ticker_docs[article.ticker].append(protect_bigrams(text, protected=PROTECTED_WORDS["extra_words"]))

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

    # free text query
    characteristics_doc = protect_bigrams(desired_characteristics, protected=PROTECTED_WORDS["extra_words"])
    characteristics_doc = characteristics_doc.replace('"', "").replace("-", " ").lower().strip()

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

        print("\n========== SVD SEARCH RESULTS ==========")

        # print top results
        sorted_tickers_idx = np.argsort(similarities)[::-1]

        recommendations_ind = []
        for idx in sorted_tickers_idx:
            if tickers[idx] not in user_portfolio:
                recommendations_ind.append(idx)

        recommendations_ind = recommendations_ind[:4]

        print("\nTop Stock Recommendations:")
        for idx in recommendations_ind:
            print(f"{tickers[idx]} -> similarity {similarities[idx]:.4f}")

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
    top_results = _enrich_with_yfinance(top_results)

    return top_results


def explain_recommendation(idx, q=[], d=[], tickers=[], vectorizer=None, svd=None, top_k_dims=3):

    print(f"\n--- {tickers[idx]} ---")

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

        print(f"\nDimension {dim}")
        print(f"  Query value: {q[dim]:.4f}")
        print(f"  Stock value: {d[dim]:.4f}")
        print(f"  Product: {products[dim]:.4f}")
        print(f"  Relationship: {relation}")
        print(f"  Positive terms: {pos_terms}")
        print(f"  Negative terms: {neg_terms}")


def get_recommendation_desc(ticker, max_articles=25):
    # take 'max_articles' number of most recent articles for ticker
    articles = (
        Article.query.filter_by(ticker=ticker.upper().strip()).order_by(Article.id.desc()).limit(max_articles).all()
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
                headline_hits[risk_type].append(article.headline)

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
    top_risks = [r for r, _ in sort_signals[:2]]

    bullets = []
    details = []
    for i, risk in enumerate(top_risks):
        keywords = list(dict.fromkeys(keyword_hits.get(risk, [])))[:2]
        if keywords:
            keyword_text = " and ".join(keywords)
            # First bullet includes news reference, subsequent ones don't
            if i == 0:
                bullet = f"{risk} due to {keyword_text} risk signals in recent news coverage"
            else:
                bullet = f"{risk} due to {keyword_text} risk signals"
        else:
            if i == 0:
                bullet = f"Susceptible to {risk} based on recent news coverage."
            else:
                bullet = f"Susceptible to {risk}."
        bullets.append(bullet)
        details.append({
            "bullet": bullet,
            "headlines": headline_hits.get(risk, []),
        })

    return {"bullets": bullets, "details": details}
