from collections import defaultdict
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from models import Article, RiskData
import numpy as np
from nltk.metrics.distance import edit_distance

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_directory))


def get_fitted_svd(
    ticker_docs={},
    query_doc={},
    vectorizer=TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3), min_df=10),
    n_components=30,
):

    tfidf_matrix = vectorizer.fit_transform(list(ticker_docs.values()))

    max_allowed_components = min(tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)

    k = min(n_components, max_allowed_components) if max_allowed_components > 1 else 1

    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit_transform(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()

    for i, component in enumerate(svd.components_):
        # positive terms
        top_indices = component.argsort()[-20:][::-1]  # top k words
        top_words = [feature_names[j] for j in top_indices]

        # negative terms
        top_neg_ind = component.argsort()[:10]
        top_neg_words = [feature_names[j] for j in top_neg_ind]

        print(f"\nDimension {i}")
        print("  Positive:", ", ".join(top_words))
        print("  Negative:", ", ".join(top_neg_words))

    return svd
