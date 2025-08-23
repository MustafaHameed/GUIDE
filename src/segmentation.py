"""Student segmentation using clustering algorithms.

References
----------
- KMeans documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- AgglomerativeClustering documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
"""

from __future__ import annotations
import os

# Set environment variable BEFORE importing scikit-learn
if os.name == 'nt':  # Windows
    os.environ['OMP_NUM_THREADS'] = '2'

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150


def save_table(table: pd.DataFrame, name: str, directory: Path) -> None:
    """Save a table to disk."""
    directory.mkdir(exist_ok=True)
    table.to_csv(directory / name, index=False)


def save_figure(ax: plt.Axes, name: str, directory: Path) -> None:
    """Tighten layout, save and close a figure given an Axes."""
    directory.mkdir(exist_ok=True)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(directory / name)
    plt.close(fig)


def segment_students(csv_path: str = "student-mat.csv", n_clusters: int = 3) -> pd.DataFrame:
    """Cluster students and output summary statistics.

    Parameters
    ----------
    csv_path : str, default 'student-mat.csv'
        Path to the student performance CSV file.
    n_clusters : int, default 3
        Number of clusters to form.

    Returns
    -------
    pandas.DataFrame
        Summary statistics for each algorithm and cluster.
    """

    df = pd.read_csv(csv_path)

    # Use only numeric features excluding the final grade for clustering
    features = df.select_dtypes(include="number").drop(columns=["G3"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    algorithms = {
        "kmeans": KMeans(n_clusters=n_clusters, random_state=0, n_init=10),  # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        # n_init set to explicit integer for compatibility with older scikit-learn
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),  # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    }

    summaries: list[pd.DataFrame] = []
    fig_dir = Path("figures")
    table_dir = Path("tables")

    for name, model in algorithms.items():
        labels = model.fit_predict(X_scaled)
        cluster_col = f"cluster_{name}"
        df[cluster_col] = labels

        # Summary statistics for the final grade per cluster
        summary = (
            df.groupby(cluster_col)["G3"].agg(["count", "mean"]).rename(
                columns={"count": "count", "mean": "avg_G3"}
            )
        )
        summary["algorithm"] = name
        summary["cluster"] = summary.index
        summaries.append(summary.reset_index(drop=True))

        # Plot average grade per cluster
        ax = sns.barplot(x=cluster_col, y="G3", data=df, estimator="mean")
        ax.set(xlabel="Cluster", ylabel="Average G3", title=f"{name.title()} Clusters")
        save_figure(ax, f"segmentation_{name}.png", fig_dir)

    summary_df = pd.concat(summaries, ignore_index=True)[
        ["algorithm", "cluster", "count", "avg_G3"]
    ]
    save_table(summary_df, "segmentation_summary.csv", table_dir)
    return summary_df


if __name__ == "__main__":
    segment_students()
