"""Exploratory data analysis utilities for the student performance dataset.

This module exposes :func:`run_eda` which generates a collection of tables and
figures describing the dataset.  The previous iteration of this module executed
these steps at import time; the functionality is now wrapped in a function so it
can be invoked programmatically by other modules (e.g. a data workflow script).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd
df = pd.read_csv("student-mat.csv")
print(df.columns)
print(df.head())
sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def _save_table(table: pd.DataFrame, name: str, directory: Path) -> None:
    """Save a table to ``directory`` ensuring the folder exists."""

    directory.mkdir(exist_ok=True)
    table.to_csv(directory / name)


def _save_figure(ax: plt.Axes, name: str, directory: Path) -> None:
    """Tighten layout, save and close a figure given an ``Axes`` instance."""

    directory.mkdir(exist_ok=True)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(directory / name)
    print(f"Saved figure: {directory / name}")  # Add this line
    plt.close(fig)


def run_eda(
    df: pd.DataFrame,
    fig_dir: str | Path = "figures",
    table_dir: str | Path = "tables",
) -> None:
    """Generate exploratory data analysis artifacts.

    Parameters
    ----------
    df:
        The full dataset including the ``G3`` final grade column.
    fig_dir, table_dir:
        Output directories for the generated figures and tables.  Directories
        are created if they do not already exist.
    """

    print(df.columns)
    print(df.head())

    fig_dir = Path(fig_dir)
    table_dir = Path(table_dir)

    # Summary statistics and additional tables
    summary = df.describe(include="all")
    print(summary)
    _save_table(summary, "summary.csv", table_dir)

    group_tables = {
        "grade_by_sex.csv": df.groupby("sex")["G3"].agg(["count", "mean"]),
        "grade_by_studytime.csv": df.groupby("studytime")["G3"].agg(["count", "mean"]),
    }

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    group_tables["correlation_matrix.csv"] = corr

    for name, table in group_tables.items():
        _save_table(table, name, table_dir)

    # Plots
    try:
        ax = sns.histplot(df["G3"], bins=20, kde=True)
        ax.set(xlabel="Final Grade (G3)", ylabel="Count")
        _save_figure(ax, "g3_distribution.png", fig_dir)
    except Exception as e:
        print(f"Failed to create g3_distribution.png: {e}")

    ax = sns.boxplot(data=df, x="sex", y="G3")
    ax.set(xlabel="Sex", ylabel="Final Grade (G3)")
    _save_figure(ax, "g3_by_sex.png", fig_dir)

    ax = sns.regplot(data=df, x="studytime", y="G3", scatter_kws={"s": 20})
    ax.set(xlabel="Weekly Study Time", ylabel="Final Grade (G3)")
    _save_figure(ax, "studytime_vs_g3.png", fig_dir)

    # Distribution of all grade columns
    grades_long = df[["G1", "G2", "G3"]].melt(
        var_name="grade", value_name="score"
    )
    ax = sns.histplot(
        data=grades_long,
        x="score",
        hue="grade",
        bins=20,
        element="step",
        stat="density",
        common_norm=False,
    )
    ax.set(xlabel="Grade", ylabel="Density")
    _save_figure(ax, "grades_distribution.png", fig_dir)

    ax = sns.regplot(data=df, x="absences", y="G3", scatter_kws={"s": 20})
    ax.set(xlabel="Number of Absences", ylabel="Final Grade (G3)")
    _save_figure(ax, "absences_vs_g3.png", fig_dir)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(corr, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.5})
    _save_figure(ax, "correlation_heatmap.png", fig_dir)

    g = sns.pairplot(
        df[["G1", "G2", "G3"]],
        kind="reg",
        diag_kind="kde",
        plot_kws={"scatter_kws": {"s": 20}},
    )
    g.fig.tight_layout()
    g.fig.savefig(fig_dir / "grades_pairplot.png")
    plt.close(g.fig)

    # Figures for the paper's main exploratory analysis
    for grade in ["G1", "G2", "G3"]:
        ax = sns.histplot(df[grade], bins=20, kde=True)
        ax.axvline(10, color="k", linestyle="--")
        ax.set(xlabel=f"{grade} Score", ylabel="Count")
        _save_figure(ax, f"eda_hist_{grade.lower()}.png", fig_dir)

    corr_subset = df[["G1", "G2", "G3", "absences", "failures", "studytime"]].corr()
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(corr_subset, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.5})
    _save_figure(ax, "eda_corr_heatmap.png", fig_dir)

    ax = sns.boxplot(data=df, x="sex", y="G3")
    ax.set(xlabel="Sex", ylabel="Final Grade (G3)")
    _save_figure(ax, "eda_g3_by_sex_box.png", fig_dir)

    ax = sns.boxplot(data=df, x="school", y="G3")
    ax.set(xlabel="School", ylabel="Final Grade (G3)")
    _save_figure(ax, "eda_g3_by_school_box.png", fig_dir)

    outcome_df = df.assign(outcome=lambda d: d["G3"].ge(10).map({True: "pass", False: "fail"}))
    ax = sns.boxplot(data=outcome_df, x="outcome", y="absences")
    ax.set(xlabel="Outcome", ylabel="Number of Absences")
    _save_figure(ax, "eda_absences_by_outcome.png", fig_dir)

    ax = sns.regplot(
        data=df, x="studytime", y="G3", lowess=True, scatter_kws={"s": 20}
    )
    ax.set(xlabel="Weekly Study Time", ylabel="Final Grade (G3)")
    _save_figure(ax, "eda_studytime_vs_g3.png", fig_dir)

    ax = sns.regplot(data=df, x="G1", y="G3", scatter_kws={"s": 20})
    ax.set(xlabel="First Period Grade (G1)", ylabel="Final Grade (G3)")
    _save_figure(ax, "eda_g1_vs_g3_scatter.png", fig_dir)


__all__ = ["run_eda"]

# Add this execution block
if __name__ == "__main__":
    # Data is already loaded at the module level as 'df'
    print("Running EDA with student data...")
    run_eda(df)
    print("EDA complete. Check the 'figures' and 'tables' directories for output.")



