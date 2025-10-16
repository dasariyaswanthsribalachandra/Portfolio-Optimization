from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def pca_kmeans_diversification(returns: pd.DataFrame, n_clusters: int = 5, random_state: int = 42):
    if returns.shape[1] < 2:
        return {0: list(returns.columns)}, [0]*returns.shape[1], None, None
    X = returns.cov().fillna(0.0).values
    n_comp = min(returns.shape[1], 5)
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_p = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=min(n_clusters, returns.shape[1]), n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_p)
    clusters = {i: [] for i in range(int(labels.max())+1)}
    for idx, col in enumerate(returns.columns):
        clusters[int(labels[idx])].append(col)
    return clusters, labels, pca, kmeans
