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
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3), min_df=10),
    embeddings=np.zeros((1, 1)),
    n_components=30,
):


    max_allowed_components = min(embeddings.shape[0] - 1, embeddings.shape[1] - 1)

    k = min(n_components, max_allowed_components) if max_allowed_components > 1 else 1

    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit_transform(embeddings)
    
    # feature_names = vectorizer.get_feature_names_out()

    # dimensions are ordered by singular value in descending order
    # print(f"\nExplained variance ratio {svd.singular_values_}")

    # for i, component in enumerate(svd.components_):
        # positive terms
        # top_indices = component.argsort()[-20:][::-1]  # top 20 positive words
        # top_words = [feature_names[j] for j in top_indices]

        # negative terms
        # top_neg_ind = component.argsort()[:10]  # top 10 negative words
        # top_neg_words = [feature_names[j] for j in top_neg_ind]

        # print(f"\nDimension {i}")
        # print("  Positive:", ", ".join(top_words))
        # print("  Negative:", ", ".join(top_neg_words))

    return svd
