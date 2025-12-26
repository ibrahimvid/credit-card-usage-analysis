import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "CC_GENERAL.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


@dataclass
class ClusteringResults:
    kmeans: KMeans
    hierarchical: AgglomerativeClustering
    features_scaled: np.ndarray
    kmeans_labels: np.ndarray
    hierarchical_labels: np.ndarray
    kmeans_silhouette: float
    kmeans_calinski: float
    kmeans_davies: float


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the credit card dataset."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop customer ID column
    - Impute numerical missing values with median
    """
    df = df.copy()
    cust_id_cols = [c for c in df.columns if c.upper().startswith("CUST")]
    df = df.drop(columns=cust_id_cols, errors="ignore")
    for col in df.columns:
        if df[col].dtype != "O":
            df[col] = df[col].fillna(df[col].median())
    return df


def scale_features(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Standard-scale numeric features."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df.values)
    return features_scaled, scaler


def train_kmeans(features: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> KMeans:
    """Train a KMeans clustering model."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(features)
    return kmeans


def train_hierarchical(features: np.ndarray, n_clusters: int = 3) -> AgglomerativeClustering:
    """Train an Agglomerative (hierarchical) clustering model."""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical.fit(features)
    return hierarchical


def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute common clustering metrics:
    - Silhouette score (higher is better, [-1, 1])
    - Calinski-Harabasz index (higher is better)
    - Davies-Bouldin index (lower is better)
    """
    sil = silhouette_score(features, labels)
    calinski = calinski_harabasz_score(features, labels)
    davies = davies_bouldin_score(features, labels)
    return sil, calinski, davies


def run_pipeline(n_clusters: int = 3) -> ClusteringResults:
    """
    End-to-end pipeline:
    - Load and clean data
    - Scale features
    - Train KMeans and Agglomerative models
    - Evaluate KMeans with clustering metrics
    """
    df_raw = load_data()
    df_clean = clean_data(df_raw)

    features_scaled, _ = scale_features(df_clean)

    kmeans = train_kmeans(features_scaled, n_clusters=n_clusters)
    hierarchical = train_hierarchical(features_scaled, n_clusters=n_clusters)

    kmeans_labels = kmeans.labels_
    hierarchical_labels = hierarchical.labels_

    sil, calinski, davies = evaluate_clustering(features_scaled, kmeans_labels)

    return ClusteringResults(
        kmeans=kmeans,
        hierarchical=hierarchical,
        features_scaled=features_scaled,
        kmeans_labels=kmeans_labels,
        hierarchical_labels=hierarchical_labels,
        kmeans_silhouette=sil,
        kmeans_calinski=calinski,
        kmeans_davies=davies,
    )


def plot_cluster_size_distribution(labels: np.ndarray, title: str, save_path: str) -> None:
    """Simple bar chart of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts)
    plt.xlabel("Cluster")
    plt.ylabel("Number of customers")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(results: ClusteringResults) -> None:
    """Persist basic artifacts for inspection."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    metrics_path = os.path.join(MODELS_DIR, "clustering_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("KMeans clustering evaluation\n")
        f.write(f"Silhouette score:        {results.kmeans_silhouette:.3f}\n")
        f.write(f"Calinski-Harabasz score: {results.kmeans_calinski:.1f}\n")
        f.write(f"Davies-Bouldin score:    {results.kmeans_davies:.3f}\n")

    plot_path = os.path.join(MODELS_DIR, "cluster_size_distribution.png")
    plot_cluster_size_distribution(
        results.kmeans_labels,
        title="Customer count per KMeans cluster",
        save_path=plot_path,
    )


def main() -> None:
    """
    Convenience entry point for running the pipeline from the command line.
    Example:
        python -m src.pipeline
    """
    results = run_pipeline(n_clusters=3)
    save_results(results)
    print("KMeans silhouette score:", round(results.kmeans_silhouette, 3))
    print("KMeans Calinski-Harabasz:", round(results.kmeans_calinski, 1))
    print("KMeans Davies-Bouldin:", round(results.kmeans_davies, 3))


if __name__ == "__main__":
    main()

