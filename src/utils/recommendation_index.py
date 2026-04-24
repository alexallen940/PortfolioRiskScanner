from collections import defaultdict
import traceback

from models import Article, RiskData
from services.svd import get_fitted_svd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.load_from_db import (
    _load_article_link_lookup,
    _load_company_metadata,
    _load_company_metadata_from_articles,
    _load_company_metadata_from_dataset,
)


class RecommendationIndex:

    def __init__(self):
        self.tickers = []
        self.ticker_docs = {}
        self.vectorizer = None
        self.tfidf_matrix = None
        self.svd = None
        self.doc_repr = None
        self.unigram_features = []
        self.feature_names = None
        self.ticker_article_rows = {}

        self.risk_by_ticker = {}
        self.risk_scores = {}
        self.min_raw_score = 0.0
        self.max_raw_score = 0.0

        self.company_metadata = {}
        self.article_link_lookup = {}
        self._built = False

    def build(
        self,
        max_features=5000,
        ngram_range=(1, 3),
        min_df=10,
        n_components=20,
    ):
        try:
            ticker_docs = defaultdict(list)
            ticker_article_rows = defaultdict(list)

            for article in Article.query.all():
                ticker_docs[article.ticker.upper().strip()].append(f"{article.headline} {article.summary}")
                ticker_article_rows[article.ticker.upper().strip()].append(article)

            ticker_docs = {
                ticker: " ".join(texts).replace('"', "").replace("-", " ").lower().strip()
                for ticker, texts in ticker_docs.items()
            }

            tickers = list(ticker_docs.keys())
            documents = [ticker_docs[t] for t in tickers]

            # tfidf
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
            )
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            unigram_features = [f for f in feature_names if "_" not in f and " " not in f]

            # svd
            svd = get_fitted_svd(vectorizer=vectorizer, embeddings=tfidf_matrix, n_components=n_components)
            doc_repr_svd = svd.transform(tfidf_matrix)

            # ticker risk data
            risk_rows = RiskData.query.all()
            risk_by_ticker = {row.ticker: row for row in risk_rows}
            risk_scores = {row.ticker: row.risk_score_1_10 for row in risk_rows}
            raw_risk_scores = [float(row.raw_risk_score) for row in risk_rows]
            min_raw = min(raw_risk_scores, default=0)
            max_raw = max(raw_risk_scores, default=0)

            # loads from db
            article_links = _load_article_link_lookup()
            company_md = _load_company_metadata()

            self.tickers = tickers
            self.ticker_docs = ticker_docs
            self.vectorizer = vectorizer
            self.tfidf_matrix = tfidf_matrix
            self.svd = svd
            self.doc_repr = doc_repr_svd
            self.unigram_features = unigram_features
            self.feature_names = feature_names
            self.risk_by_ticker = risk_by_ticker
            self.risk_scores = risk_scores
            self.min_raw_score = min_raw
            self.max_raw_score = max_raw
            self.company_metadata = company_md
            self.article_link_lookup = article_links
            self.ticker_article_rows = ticker_article_rows
            self._built = True

        except Exception:
            traceback.print_exc()

    def ensure_built(self):
        if not self._built:
            self.build()


