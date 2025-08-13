"""Student segmentation using clustering algorithms."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


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
        "kmeans": KMeans(n_clusters=n_clusters, random_state=0, n_init="auto"),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
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
