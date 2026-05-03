import traceback
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from models import Article, RiskData
from services.svd import get_fitted_svd
from utils.load_from_db import _load_article_link_lookup, _load_company_metadata
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
stop_words_path = os.path.join(BASE_DIR, "data", "stop_words.json")

with open(stop_words_path, "r") as f:
    STOP_WORDS = json.load(f)


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
        self._last_error = None

    def build(self, max_features=5000, ngram_range=(1, 3), min_df=10, n_components=20, max_df=0.95):
        try:
            ticker_docs = defaultdict(list)
            ticker_article_rows = defaultdict(list)

            for article in Article.query.all():
                key = article.ticker.upper().strip()
                ticker_docs[key].append(f"{article.headline} {article.summary}")
                ticker_article_rows[key].append(article)

            ticker_docs = {
                ticker: " ".join(texts).replace('"', "").replace("-", " ").lower().strip()
                for ticker, texts in ticker_docs.items()
            }

            tickers = list(ticker_docs.keys())
            documents = [ticker_docs[t] for t in tickers]

            # Vectorizer
            vectorizer = TfidfVectorizer(
                stop_words="english", max_features=max_features, ngram_range=ngram_range, min_df=min_df, max_df=max_df
            )

            custom_stops = (
                list(vectorizer.get_stop_words())
                + [w.lower().strip() for w in STOP_WORDS["company_names"]]
                + [w.lower().strip() for w in STOP_WORDS["extra_words"]]
                + [w.lower().strip() for w in STOP_WORDS["people_names"]]
            )

            vectorizer = TfidfVectorizer(
                stop_words=custom_stops,
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
            )

            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            unigram_features = [f for f in feature_names if "_" not in f and " " not in f]

            # svd
            svd = get_fitted_svd(vectorizer=vectorizer, embeddings=tfidf_matrix, n_components=n_components)
            doc_repr_svd = svd.transform(tfidf_matrix)

            # ticker risk data
            risk_rows = RiskData.query.all()
            raw_risk_scores = [float(row.raw_risk_score) for row in risk_rows]

            self.tickers = tickers
            self.ticker_docs = ticker_docs
            self.vectorizer = vectorizer
            self.tfidf_matrix = tfidf_matrix
            self.svd = svd
            self.doc_repr = doc_repr_svd
            self.unigram_features = unigram_features
            self.feature_names = feature_names
            self.ticker_article_rows = ticker_article_rows
            self.risk_by_ticker = {row.ticker: row for row in risk_rows}
            self.risk_scores = {row.ticker: row.risk_score_1_10 for row in risk_rows}
            self.min_raw_score = min(raw_risk_scores, default=0.0)
            self.max_raw_score = max(raw_risk_scores, default=0.0)
            self.company_metadata = _load_company_metadata()
            self.article_link_lookup = _load_article_link_lookup()
            self._built = True
            self._last_error = None
            return True

        except Exception as exc:
            traceback.print_exc()
            self._built = False
            self._last_error = exc
            return False

    def ensure_built(self):
        if not self._built:
            built = self.build()
            if not built:
                raise RuntimeError("Recommendation index is not available") from self._last_error
