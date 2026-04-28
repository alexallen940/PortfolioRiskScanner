import numpy as np
from sklearn.decomposition import TruncatedSVD


def get_fitted_svd(vectorizer=None, embeddings=np.zeros((1, 1)), n_components=30):

    max_allowed_components = min(embeddings.shape[0] - 1, embeddings.shape[1] - 1)
    k = min(n_components, max_allowed_components) if max_allowed_components > 1 else 1

    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit_transform(embeddings)
    return svd
