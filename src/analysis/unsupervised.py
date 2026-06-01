"""Unsupervised exploration: PCA, t-SNE, KMeans + silhouette, cluster profiling.

These are **exploratory** tools for the CRISP-DM *Data Understanding* phase --
patient phenotyping and structure-spotting. They do **not** predict LOS and are
not part of the supervised pipeline. Two honest caveats applied throughout:

* clinical cohorts rarely form cleanly separated clusters, so silhouette scores
  are typically low (~0.1-0.3) -- we report k by best silhouette but read it as
  "weak structure", not "natural classes";
* t-SNE preserves local neighbourhoods only -- distances/cluster sizes in the 2-D
  map are not meaningful, so it is used for visualization, never for decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config as cfg


def _prep(X: pd.DataFrame) -> np.ndarray:
    """Median-impute + standardise -- required for distance-based methods."""
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
    return pipe.fit_transform(X)


@dataclass
class PCAResult:
    coords: np.ndarray            # (n, 2)
    explained: np.ndarray         # explained variance ratio (all components)
    loadings: pd.DataFrame        # top feature loadings for PC1/PC2


def pca_view(X: pd.DataFrame, n_components: int = 10) -> PCAResult:
    Z = _prep(X)
    pca = PCA(n_components=min(n_components, X.shape[1]),
              random_state=cfg.RANDOM_STATE).fit(Z)
    coords = pca.transform(Z)[:, :2]
    loadings = pd.DataFrame(
        pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
    return PCAResult(coords, pca.explained_variance_ratio_, loadings)


def tsne_view(X: pd.DataFrame, sample: int = 3000, perplexity: int = 30) -> tuple:
    """Return (coords, sample_index). Subsamples for tractability on big cohorts."""
    idx = X.index
    if len(X) > sample:
        rng = np.random.default_rng(cfg.RANDOM_STATE)
        idx = pd.Index(rng.choice(X.index, size=sample, replace=False))
    Z = _prep(X.loc[idx])
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  random_state=cfg.RANDOM_STATE).fit_transform(Z)
    return coords, idx


@dataclass
class ClusterResult:
    k_scores: dict[int, float]    # k -> silhouette
    best_k: int
    labels: pd.Series             # cluster id per stay (best_k)


def kmeans_silhouette(X: pd.DataFrame, k_range=range(2, 7),
                      sample: int = 5000) -> ClusterResult:
    """KMeans across k; pick k by silhouette (computed on a sample for speed)."""
    Z = _prep(X)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    sil_idx = (rng.choice(len(Z), size=sample, replace=False)
               if len(Z) > sample else np.arange(len(Z)))
    scores: dict[int, float] = {}
    label_sets: dict[int, np.ndarray] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.RANDOM_STATE)
        labels = km.fit_predict(Z)
        scores[k] = float(silhouette_score(Z[sil_idx], labels[sil_idx]))
        label_sets[k] = labels
    best_k = max(scores, key=scores.get)
    return ClusterResult(scores, best_k,
                         pd.Series(label_sets[best_k], index=X.index, name="cluster"))


def cluster_profile(labels: pd.Series, y_reg: pd.Series,
                    y_clf: pd.Series) -> pd.DataFrame:
    """Per-cluster size + LOS profile -- do the phenotypes differ in stay length?"""
    df = pd.DataFrame({"cluster": labels, "los_days": y_reg, "bucket": y_clf})
    out = df.groupby("cluster").agg(
        n=("los_days", "size"),
        los_mean=("los_days", "mean"),
        los_median=("los_days", "median"),
        pct_long=("bucket", lambda s: float((s == "long").mean())),
    ).round(2)
    return out
