"""Exploratory data analysis utilities for the student performance dataset.

This module exposes :func:`run_eda` which generates a collection of tables and
figures describing the dataset.  The previous iteration of this module executed
these steps at import time; the functionality is now wrapped in a function so it
can be invoked programmatically by other modules (e.g. a data workflow script).
"""

from __future__ import annotations

from pathlib import Path

import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, contingency
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

try:
    from .utils import ensure_dir
except ImportError:  # pragma: no cover - fallback for direct execution
    from utils import ensure_dir

logger = logging.getLogger(__name__)

# Publication-quality styling
sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def _save_table(table: pd.DataFrame, name: str, directory: Path) -> None:
    """Save a table to ``directory`` ensuring the folder exists."""
    ensure_dir(directory)
    table.to_csv(directory / name)


def _save_figure(ax: plt.Axes, name: str, directory: Path) -> None:
    """Tighten layout, save and close a figure given an ``Axes`` instance."""

    ensure_dir(directory)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(directory / name, dpi=300, bbox_inches="tight")
    logger.info("Saved figure: %s", directory / name)
    plt.close(fig)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramér's V statistic for association between categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def _analyze_categorical_associations(
    df: pd.DataFrame, cat_cols: list, target_col: str = "G3"
) -> pd.DataFrame:
    """Analyze associations between categorical variables and target."""
    results = []

    # Encode categorical variables for mutual information
    df_encoded = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # Calculate mutual information with target
    if target_col in df.select_dtypes(include=["number"]).columns:
        # Regression case
        mi_scores = mutual_info_regression(
            df_encoded[cat_cols], df_encoded[target_col], random_state=42
        )
    else:
        # Classification case
        mi_scores = mutual_info_classif(
            df_encoded[cat_cols], df_encoded[target_col], random_state=42
        )

    for i, col in enumerate(cat_cols):
        # Chi-square test
        contingency_table = pd.crosstab(
            df[col], pd.cut(df[target_col], bins=3, labels=["Low", "Med", "High"])
        )
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        # Cramér's V
        cramers_v = _cramers_v(
            df[col], pd.cut(df[target_col], bins=3, labels=["Low", "Med", "High"])
        )

        results.append(
            {
                "variable": col,
                "mutual_info": mi_scores[i],
                "chi2_stat": chi2,
                "chi2_pvalue": p_value,
                "cramers_v": cramers_v,
                "unique_values": df[col].nunique(),
            }
        )

    return pd.DataFrame(results).sort_values("mutual_info", ascending=False)


def _create_categorical_correlation_matrix(
    df: pd.DataFrame, cat_cols: list
) -> pd.DataFrame:
    """Create correlation matrix for categorical variables using Cramér's V."""
    n_vars = len(cat_cols)
    corr_matrix = np.ones((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            cramers_v = _cramers_v(df[cat_cols[i]], df[cat_cols[j]])
            corr_matrix[i, j] = cramers_v
            corr_matrix[j, i] = cramers_v

    return pd.DataFrame(corr_matrix, index=cat_cols, columns=cat_cols)


def run_eda(
    df: pd.DataFrame,
    fig_dir: str | Path = "figures",
    table_dir: str | Path = "tables",
    report_dir: str | Path = "reports",
) -> None:
    """Generate exploratory data analysis artifacts.

    Parameters
    ----------
    df:
        The full dataset including the ``G3`` final grade column.
    fig_dir, table_dir, report_dir:
        Output directories for the generated figures, tables, and reports.
        Directories are created if they do not already exist.
    """

    logger.debug(df.columns)
    logger.debug(df.head())

    fig_dir = Path(fig_dir)
    table_dir = Path(table_dir)
    report_dir = Path(report_dir)

    # Separate categorical and numerical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    logger.info(
        f"Found {len(cat_cols)} categorical and {len(num_cols)} numerical variables"
    )

    # Summary statistics and additional tables
    summary = df.describe(include="all")
    logger.debug("Summary statistics:\n%s", summary)
    _save_table(summary, "eda_summary_statistics.csv", table_dir)

    group_tables = {
        "eda_grade_by_sex.csv": df.groupby("sex")["G3"].agg(["count", "mean", "std"]),
        "eda_grade_by_studytime.csv": df.groupby("studytime")["G3"].agg(
            ["count", "mean", "std"]
        ),
        "eda_grade_by_school.csv": df.groupby("school")["G3"].agg(
            ["count", "mean", "std"]
        ),
    }

    # Numerical correlations
    numeric_df = df.select_dtypes(include="number")
    numeric_corr = numeric_df.corr()
    group_tables["eda_numeric_correlation_matrix.csv"] = numeric_corr

    # Categorical variable analysis
    if cat_cols:
        cat_associations = _analyze_categorical_associations(df, cat_cols, "G3")
        group_tables["eda_categorical_feature_importance.csv"] = cat_associations

        # Categorical correlation matrix using Cramér's V
        cat_corr = _create_categorical_correlation_matrix(df, cat_cols)
        group_tables["eda_categorical_correlation_matrix.csv"] = cat_corr

    for name, table in group_tables.items():
        _save_table(table, name, table_dir)

    # =============================================================================
    # VISUALIZATIONS
    # =============================================================================

    # Target variable distribution
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.histplot(df["G3"], bins=20, kde=True, ax=ax)
        ax.axvline(
            df["G3"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {df['G3'].mean():.1f}",
        )
        ax.axvline(10, color="orange", linestyle="--", label="Pass threshold (10)")
        ax.set(
            xlabel="Final Grade (G3)",
            ylabel="Count",
            title="Distribution of Final Grades",
        )
        ax.legend()
        _save_figure(ax, "eda_target_distribution.png", fig_dir)
    except Exception as e:
        logger.error("Failed to create eda_target_distribution.png: %s", e)

    # Grade distributions comparison
    for grade in ["G1", "G2", "G3"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.histplot(df[grade], bins=20, kde=True, ax=ax)
        ax.axvline(10, color="k", linestyle="--", alpha=0.7, label="Pass threshold")
        ax.axvline(
            df[grade].mean(),
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {df[grade].mean():.1f}",
        )
        ax.set(
            xlabel=f"{grade} Score",
            ylabel="Count",
            title=f"Distribution of {grade} Scores",
        )
        ax.legend()
        _save_figure(ax, f"eda_grade_distribution_{grade.lower()}.png", fig_dir)

    # All grades distribution comparison
    grades_long = df[["G1", "G2", "G3"]].melt(var_name="grade", value_name="score")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.histplot(
        data=grades_long,
        x="score",
        hue="grade",
        bins=20,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.7,
        ax=ax,
    )
    ax.axvline(10, color="k", linestyle="--", alpha=0.5, label="Pass threshold")
    ax.set(
        xlabel="Grade Score",
        ylabel="Density",
        title="Distribution Comparison of All Grade Periods",
    )
    ax.legend()
    _save_figure(ax, "eda_all_grades_distribution.png", fig_dir)

    # Correlation heatmaps
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(numeric_corr, dtype=bool))
    ax = sns.heatmap(
        numeric_corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Numerical Variables Correlation Matrix")
    _save_figure(ax, "eda_numeric_correlation_heatmap.png", fig_dir)

    if cat_cols:
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(cat_corr, dtype=bool))
        ax = sns.heatmap(
            cat_corr,
            mask=mask,
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Categorical Variables Association Matrix (Cramér's V)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        _save_figure(ax, "eda_categorical_correlation_heatmap.png", fig_dir)

    # Feature importance plot for categorical variables
    if cat_cols:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = cat_associations.head(10)
        ax = sns.barplot(
            data=top_features,
            y="variable",
            x="mutual_info",
            hue="variable",
            palette="viridis",
            legend=False,
            ax=ax,
        )
        ax.set(
            xlabel="Mutual Information Score",
            ylabel="Categorical Variables",
            title="Feature Importance: Categorical Variables vs Final Grade",
        )
        _save_figure(ax, "eda_categorical_feature_importance.png", fig_dir)

    # Categorical variable visualizations
    outcome_df = df.assign(
        outcome=lambda d: d["G3"].ge(10).map({True: "Pass", False: "Fail"})
    )

    # Key categorical variables vs outcome
    key_cat_vars = ["sex", "school", "higher", "internet", "romantic", "activities"]
    available_vars = [var for var in key_cat_vars if var in cat_cols]

    for var in available_vars:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot
        sns.boxplot(data=df, x=var, y="G3", ax=ax1)
        ax1.axhline(10, color="red", linestyle="--", alpha=0.7, label="Pass threshold")
        ax1.set_title(f"Grade Distribution by {var.title()}")
        ax1.set_ylabel("Final Grade (G3)")
        ax1.legend()

        # Count plot with outcome
        sns.countplot(data=outcome_df, x=var, hue="outcome", ax=ax2)
        ax2.set_title(f"Pass/Fail Count by {var.title()}")
        ax2.set_ylabel("Count")

        plt.tight_layout()
        _save_figure(ax1, f"eda_grade_by_{var}_analysis.png", fig_dir)

    # Numeric relationships
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.regplot(
        data=df, x="G1", y="G3", scatter_kws={"s": 30, "alpha": 0.6}, ax=ax
    )
    ax.set(
        xlabel="First Period Grade (G1)",
        ylabel="Final Grade (G3)",
        title="Relationship between First Period and Final Grades",
    )
    _save_figure(ax, "eda_g1_vs_g3_relationship.png", fig_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.regplot(
        data=df,
        x="studytime",
        y="G3",
        lowess=True,
        scatter_kws={"s": 30, "alpha": 0.6},
        ax=ax,
    )
    ax.set(
        xlabel="Weekly Study Time",
        ylabel="Final Grade (G3)",
        title="Relationship between Study Time and Final Grades",
    )
    _save_figure(ax, "eda_studytime_vs_g3_relationship.png", fig_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.regplot(
        data=df, x="absences", y="G3", scatter_kws={"s": 30, "alpha": 0.6}, ax=ax
    )
    ax.set(
        xlabel="Number of Absences",
        ylabel="Final Grade (G3)",
        title="Relationship between Absences and Final Grades",
    )
    _save_figure(ax, "eda_absences_vs_g3_relationship.png", fig_dir)

    # Pairplot for key numerical variables
    key_num_vars = ["G1", "G2", "G3", "studytime", "absences", "age"]
    available_num_vars = [var for var in key_num_vars if var in num_cols]

    g = sns.pairplot(
        df[available_num_vars],
        kind="reg",
        diag_kind="kde",
        plot_kws={"scatter_kws": {"s": 20, "alpha": 0.6}},
    )
    g.fig.suptitle("Pairwise Relationships: Key Numerical Variables", y=1.02)
    g.fig.tight_layout()
    g.fig.savefig(fig_dir / "eda_numeric_pairplot.png", dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    # Generate narrative summary
    _generate_narrative_summary(df, cat_associations if cat_cols else None, numeric_corr, report_dir)


def _generate_narrative_summary(
    df: pd.DataFrame, 
    cat_associations: pd.DataFrame | None, 
    numeric_corr: pd.DataFrame, 
    out_dir: Path
) -> None:
    """Generate a narrative summary of the EDA findings."""
    out_dir.mkdir(exist_ok=True)
    
    # Calculate key statistics
    g3_mean = df['G3'].mean()
    g3_std = df['G3'].std()
    passing_students = (df['G3'] >= 10).sum()
    passing_rate = (passing_students / len(df)) * 100
    g1_g3_corr = df['G1'].corr(df['G3']) if 'G1' in df.columns else 0
    g2_g3_corr = df['G2'].corr(df['G3']) if 'G2' in df.columns else 0
    avg_studytime = df['studytime'].mean() if 'studytime' in df.columns else 0
    avg_absences = df['absences'].mean() if 'absences' in df.columns else 0
    
    # Build narrative text
    report = f"""# Student Performance Dataset Analysis

## Dataset Overview
The dataset contains {len(df)} students with {len(df.columns)} features.

## Grade Distribution Analysis
- Average final grade (G3): {g3_mean:.2f}
- Standard deviation: {g3_std:.2f}
- Students with G3 >= 10 (passing): {passing_students} ({passing_rate:.1f}%)

## Key Findings
{chr(8226)} The correlation between G1 and G3 is {g1_g3_corr:.3f}
{chr(8226)} The correlation between G2 and G3 is {g2_g3_corr:.3f}
{chr(8226)} Students study on average {avg_studytime:.1f} hours per week
{chr(8226)} Average absences: {avg_absences:.1f}

## Recommendations
Based on the analysis, early grades (G1, G2) are strong predictors of final performance.
Intervention programs should focus on students with low G1/G2 scores.
"""

    report_path = out_dir / "eda_narrative_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Generated narrative summary: {report_path}")


__all__ = ["run_eda"]

# Add this execution block
if __name__ == "__main__":
    df = pd.read_csv("student-mat.csv")
    logger.info("Running EDA with student data...")
    run_eda(df)
    logger.info(
        "EDA complete. Check the 'figures' and 'tables' directories for output."
    )
