import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from services.svd import get_fitted_svd
from models import Article
import json
import os

# load the risk word bank
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
json_path = os.path.join(BASE_DIR, "data", "risk_word_bank.json")

with open(json_path, "r") as f:
    RISK_KEYWORDS = json.load(f)

json_path = os.path.join(BASE_DIR, "data", "protected_words.json")

with open(json_path, "r") as f:
    PROTECTED_WORDS = json.load(f)


def protect_bigrams(text, protected):
    for bigram in protected:
        text = re.sub(rf"\b{bigram}\b", bigram.replace(" ", "_"), text, flags=re.IGNORECASE)
    return text


def get_stock_recommendations(
    user_portfolio,
    top_k=10,
    vectorizer=TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3), min_df=10),
    use_svd=True,
    n_components=20,
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

    # has 503 rows (one entry for each stock) and a certain number of words or phrases for columns
    tfidf_matrix = vectorizer.fit_transform(documents)

    portfolio_texts = []
    for ticker in user_portfolio:
        if ticker in ticker_docs:
            portfolio_texts.append(ticker_docs[ticker])

    # case when no valid portfolio tickers were found
    if not portfolio_texts:
        return []

    # combine all portfolio ticker docs into one big doc
    portfolio_doc = " ".join(portfolio_texts).replace('"', "").replace("-", " ").lower().strip()

    # same TF-IDF space as the S&P 500
    portfolio_vector = vectorizer.transform([portfolio_doc])

    if use_svd:
        svd = get_fitted_svd(ticker_docs, portfolio_doc, vectorizer, n_components)
        doc_repr = svd.transform(tfidf_matrix)
        query_repr = svd.transform(portfolio_vector)
    else:
        doc_repr = tfidf_matrix
        query_repr = portfolio_vector

    # length 503 (for each stock in the S&P 500's similarity score to portfolio)
    similarities = cosine_similarity(query_repr, doc_repr).flatten()

    results = []
    for i, score in enumerate(similarities):
        ticker = tickers[i]
        if ticker not in user_portfolio:
            results.append({"ticker": ticker, "similarity": float(score)})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def get_recommendation_desc(ticker, max_articles=25):
    # take 'max_articles' number of most recent articles for ticker
    articles = (
        Article.query.filter_by(ticker=ticker.upper().strip()).order_by(Article.id.desc()).limit(max_articles).all()
    )

    # combine all headlines and summary into one string
    combined_text = " ".join(f"{a.headline} {a.summary}".replace("-", " ").lower() for a in articles)

    # counts number of times each risk signal appears
    risk_counts = Counter()

    # keeps track of which keyword within the risk signal was similar
    keyword_hits = {}

    # count keyword matches
    for risk_type, keywords in RISK_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", combined_text):
                risk_counts[risk_type] += 1

                if risk_type not in keyword_hits:
                    keyword_hits[risk_type] = []
                keyword_hits[risk_type].append(keyword)

    # the case where no keywords match
    if not risk_counts:
        return [
            "No apparent risk themes based on recent news.",
            "Need more info to generate summary.",
        ]

    # top 2 risk types
    sort_signals = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)
    top_risks = []
    for r, ct in sort_signals[:2]:
        top_risks.append(r)

    bullets = []
    for risk in top_risks:
        # top 2 keywords
        keywords = keyword_hits.get(risk, [])[:2]

        if keywords:
            keyword_text = " and ".join(keywords)
            bullets.append(f"Signals of {risk} linked to {keyword_text}.")
        else:
            bullets.append(f"Susceptible to {risk} based on recent news coverage.")
    return bullets
