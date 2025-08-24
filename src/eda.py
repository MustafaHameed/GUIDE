"""Exploratory data analysis utilities for the student performance dataset.

This module exposes :func:`run_eda` which generates a collection of tables and
figures describing the dataset.  The previous iteration of this module executed
these steps at import time; the functionality is now wrapped in a function so it
can be invoked programmatically by other modules (e.g. a data workflow script).
"""

from __future__ import annotations

from pathlib import Path

import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

try:
    from .utils import ensure_dir
except ImportError:  # pragma: no cover - fallback for direct execution
    from utils import ensure_dir

logger = logging.getLogger(__name__)

# Publication-quality styling standards
sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (8, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "patch.linewidth": 1,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1
})


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramer's V statistic for categorical-categorical association."""
    crosstab = pd.crosstab(x, y)
    chi2 = chi2_contingency(crosstab)[0]
    n = crosstab.sum().sum()
    phi2 = chi2 / n
    r, k = crosstab.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Calculate correlation ratio (eta) for categorical-numeric association."""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    
    for i in range(cat_num):
        cat_measures = measurements[fcat == i]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    
    if denominator == 0:
        return 0
    else:
        return np.sqrt(numerator / denominator)


def _calculate_categorical_importance(df: pd.DataFrame, target_col: str = "G3") -> pd.DataFrame:
    """Calculate feature importance for categorical variables."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    importance_data = []
    target = df[target_col]
    
    # Convert target to binary for classification-based metrics
    target_binary = (target >= 10).astype(int)  # Pass/fail threshold
    
    for col in categorical_cols:
        if col in df.columns:
            # Encode categorical variable
            le = LabelEncoder()
            encoded_feature = le.fit_transform(df[col].astype(str))
            
            # Mutual information for classification (pass/fail)
            mi_class = mutual_info_classif(encoded_feature.reshape(-1, 1), target_binary, random_state=42)[0]
            
            # Mutual information for regression (actual grade)
            mi_reg = mutual_info_regression(encoded_feature.reshape(-1, 1), target, random_state=42)[0]
            
            # Chi-square test for independence
            try:
                contingency_table = pd.crosstab(df[col], target_binary)
                chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
                chi2_importance = chi2_stat / contingency_table.sum().sum()  # Normalized chi-square
            except:
                chi2_importance = 0
                p_val = 1.0
            
            # Correlation ratio (eta-squared)
            eta_squared = _correlation_ratio(df[col], target) ** 2
            
            importance_data.append({
                'feature': col,
                'mutual_info_classification': mi_class,
                'mutual_info_regression': mi_reg,
                'chi_square_normalized': chi2_importance,
                'chi_square_p_value': p_val,
                'eta_squared': eta_squared,
                'overall_importance': (mi_class + mi_reg + eta_squared) / 3  # Combined score
            })
    
    importance_df = pd.DataFrame(importance_data)
    return importance_df.sort_values('overall_importance', ascending=False)


def _calculate_categorical_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for categorical variables using Cramer's V."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    n_vars = len(categorical_cols)
    
    if n_vars < 2:
        return pd.DataFrame()
    
    corr_matrix = np.zeros((n_vars, n_vars))
    
    for i, var1 in enumerate(categorical_cols):
        for j, var2 in enumerate(categorical_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                cramers_v = _cramers_v(df[var1], df[var2])
                corr_matrix[i, j] = cramers_v
                corr_matrix[j, i] = cramers_v
    
    return pd.DataFrame(corr_matrix, index=categorical_cols, columns=categorical_cols)


def _generate_eda_summary(df: pd.DataFrame, categorical_importance: pd.DataFrame, 
                         categorical_correlations: pd.DataFrame) -> str:
    """Generate a narrative summary of EDA findings."""
    summary = []
    
    # Dataset overview
    n_samples, n_features = df.shape
    n_categorical = len(df.select_dtypes(include=['object']).columns)
    n_numerical = len(df.select_dtypes(include=['number']).columns)
    
    summary.append("# Exploratory Data Analysis Summary\n")
    summary.append(f"## Dataset Overview")
    summary.append(f"- **Total samples**: {n_samples:,}")
    summary.append(f"- **Total features**: {n_features}")
    summary.append(f"- **Categorical features**: {n_categorical}")
    summary.append(f"- **Numerical features**: {n_numerical}")
    
    # Grade distribution insights
    if 'G3' in df.columns:
        mean_grade = df['G3'].mean()
        pass_rate = (df['G3'] >= 10).mean() * 100
        summary.append(f"- **Average final grade (G3)**: {mean_grade:.1f}")
        summary.append(f"- **Pass rate (≥10)**: {pass_rate:.1f}%\n")
    
    # Categorical feature importance
    if not categorical_importance.empty:
        summary.append("## Key Categorical Features")
        summary.append("The most important categorical features for predicting student performance:\n")
        
        top_features = categorical_importance.head(5)
        for _, row in top_features.iterrows():
            feat = row['feature']
            importance = row['overall_importance']
            summary.append(f"- **{feat}**: Importance score {importance:.3f}")
            
            # Add specific insights based on feature name
            if feat == 'school':
                schools = df[feat].value_counts()
                summary.append(f"  - {len(schools)} schools in dataset")
            elif feat == 'sex':
                sex_dist = df[feat].value_counts()
                summary.append(f"  - Gender distribution: {dict(sex_dist)}")
            elif feat in ['Mjob', 'Fjob']:
                job_dist = df[feat].value_counts()
                top_job = job_dist.index[0]
                summary.append(f"  - Most common: {top_job} ({job_dist.iloc[0]} students)")
        
        summary.append("")
    
    # Correlation insights
    if not categorical_correlations.empty and categorical_correlations.shape[0] > 1:
        summary.append("## Categorical Variable Relationships")
        
        # Find highest correlations (excluding diagonal)
        corr_values = categorical_correlations.values
        np.fill_diagonal(corr_values, 0)
        max_corr_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
        max_corr_val = corr_values[max_corr_idx]
        
        if max_corr_val > 0.1:  # Only report meaningful correlations
            var1 = categorical_correlations.index[max_corr_idx[0]]
            var2 = categorical_correlations.columns[max_corr_idx[1]]
            summary.append(f"- Strongest categorical association: **{var1}** ↔ **{var2}** (Cramer's V = {max_corr_val:.3f})")
        
        # Count meaningful associations
        meaningful_corrs = np.sum(corr_values > 0.2)
        summary.append(f"- Number of strong categorical associations (Cramer's V > 0.2): {meaningful_corrs}")
        summary.append("")
    
    # Recommendations
    summary.append("## Key Insights and Recommendations")
    
    if not categorical_importance.empty:
        top_feature = categorical_importance.iloc[0]['feature']
        summary.append(f"- **{top_feature}** shows the strongest relationship with academic performance")
        
        if 'failures' in df.columns:
            failure_impact = df.groupby('failures')['G3'].mean()
            if len(failure_impact) > 1:
                summary.append(f"- Students with no previous failures average {failure_impact[0]:.1f} points")
                summary.append(f"- Students with failures average {failure_impact[failure_impact.index > 0].mean():.1f} points")
    
    if 'absences' in df.columns:
        high_absence = df['absences'] > df['absences'].quantile(0.75)
        avg_grade_low_abs = df[~high_absence]['G3'].mean()
        avg_grade_high_abs = df[high_absence]['G3'].mean()
        summary.append(f"- Students with low absences average {avg_grade_low_abs:.1f} points vs {avg_grade_high_abs:.1f} for high absences")
    
    return '\n'.join(summary)


def _save_table(table: pd.DataFrame, name: str, directory: Path) -> None:
    """Save a table to ``directory`` ensuring the folder exists."""

    ensure_dir(directory)
    table.to_csv(directory / name)


def _save_figure(ax: plt.Axes, name: str, directory: Path) -> None:
    """Tighten layout, save and close a figure given an ``Axes`` instance."""

    ensure_dir(directory)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(directory / name)
    logger.info("Saved figure: %s", directory / name)
    plt.close(fig)


def run_eda(
    df: pd.DataFrame,
    fig_dir: str | Path = "figures",
    table_dir: str | Path = "tables",
    report_dir: str | Path = "reports",
) -> None:
    """Generate comprehensive exploratory data analysis artifacts.

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

    # Summary statistics and additional tables
    summary = df.describe(include="all")
    logger.debug("Summary statistics:\n%s", summary)
    _save_table(summary, "eda_summary_statistics.csv", table_dir)

    # Traditional groupby tables
    group_tables = {
        "eda_grade_by_sex.csv": df.groupby("sex")["G3"].agg(["count", "mean", "std"]),
        "eda_grade_by_studytime.csv": df.groupby("studytime")["G3"].agg(["count", "mean", "std"]),
        "eda_grade_by_school.csv": df.groupby("school")["G3"].agg(["count", "mean", "std"]),
    }

    # Numeric correlations
    numeric_df = df.select_dtypes(include="number")
    numeric_corr = numeric_df.corr()
    group_tables["eda_numeric_correlation_matrix.csv"] = numeric_corr

    # NEW: Categorical feature importance analysis
    logger.info("Calculating categorical feature importance...")
    categorical_importance = _calculate_categorical_importance(df)
    group_tables["eda_categorical_feature_importance.csv"] = categorical_importance

    # NEW: Categorical correlations using Cramer's V
    logger.info("Calculating categorical correlations...")
    categorical_correlations = _calculate_categorical_correlations(df)
    if not categorical_correlations.empty:
        group_tables["eda_categorical_correlations.csv"] = categorical_correlations

    # Save all tables
    for name, table in group_tables.items():
        _save_table(table, name, table_dir)

    # === BASIC DISTRIBUTION PLOTS ===
    logger.info("Generating distribution plots...")
    
    # Main target variable distribution
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.histplot(df["G3"], bins=20, kde=True, ax=ax)
        ax.axvline(10, color='red', linestyle='--', alpha=0.7, label='Pass threshold')
        ax.set(xlabel="Final Grade (G3)", ylabel="Count", title="Distribution of Final Grades")
        ax.legend()
        _save_figure(ax, "eda_final_grade_distribution.png", fig_dir)
    except Exception as e:
        logger.error("Failed to create final grade distribution: %s", e)

    # All grades distribution comparison
    grades_long = df[["G1", "G2", "G3"]].melt(var_name="grade", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.histplot(
        data=grades_long, x="score", hue="grade", bins=20, element="step",
        stat="density", common_norm=False, ax=ax
    )
    ax.set(xlabel="Grade", ylabel="Density", title="Grade Distribution Across Periods")
    _save_figure(ax, "eda_grades_distribution_comparison.png", fig_dir)

    # === CATEGORICAL ANALYSIS PLOTS ===
    logger.info("Generating categorical analysis plots...")
    
    # Feature importance visualization
    if not categorical_importance.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = categorical_importance.head(10)
        sns.barplot(data=top_features, y='feature', x='overall_importance', ax=ax)
        ax.set(xlabel='Importance Score', title='Categorical Feature Importance for Academic Performance')
        _save_figure(ax, "eda_categorical_feature_importance.png", fig_dir)

    # Categorical correlations heatmap
    if not categorical_correlations.empty and categorical_correlations.shape[0] > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(categorical_correlations.values, dtype=bool))
        sns.heatmap(categorical_correlations, mask=mask, cmap="viridis", center=0, 
                   square=True, cbar_kws={"shrink": 0.8}, ax=ax,
                   annot=True, fmt='.2f')
        ax.set_title("Categorical Variables Correlation Matrix (Cramer's V)")
        _save_figure(ax, "eda_categorical_correlation_heatmap.png", fig_dir)

    # === CATEGORICAL VS TARGET ANALYSIS ===
    logger.info("Generating categorical vs target plots...")
    
    # Key categorical variables analysis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    important_categoricals = ['sex', 'school', 'Mjob', 'Fjob', 'higher', 'internet', 'romantic']
    available_categoricals = [col for col in important_categoricals if col in categorical_cols]
    
    for col in available_categoricals[:6]:  # Limit to avoid too many plots
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.boxplot(data=df, x=col, y="G3", ax=ax)
        ax.set(xlabel=col.title(), ylabel="Final Grade (G3)", 
               title=f"Final Grade Distribution by {col.title()}")
        plt.xticks(rotation=45)
        _save_figure(ax, f"eda_grade_by_{col}.png", fig_dir)

    # === NUMERIC CORRELATION ANALYSIS ===
    logger.info("Generating numeric correlation plots...")
    
    # Comprehensive numeric correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(numeric_corr.values, dtype=bool))
    sns.heatmap(numeric_corr, mask=mask, cmap="RdBu_r", center=0, square=True, 
               cbar_kws={"shrink": 0.8}, ax=ax, annot=True, fmt='.2f')
    ax.set_title("Numeric Variables Correlation Matrix")
    _save_figure(ax, "eda_numeric_correlation_heatmap.png", fig_dir)

    # Key numeric relationships
    key_numeric_pairs = [
        ("G1", "G3", "First Period vs Final Grade"),
        ("G2", "G3", "Second Period vs Final Grade"), 
        ("studytime", "G3", "Study Time vs Final Grade"),
        ("absences", "G3", "Absences vs Final Grade"),
        ("failures", "G3", "Previous Failures vs Final Grade")
    ]
    
    for x_col, y_col, title in key_numeric_pairs:
        if x_col in df.columns and y_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={"s": 30, "alpha": 0.6}, ax=ax)
            ax.set(xlabel=x_col, ylabel=y_col, title=title)
            _save_figure(ax, f"eda_{x_col}_vs_{y_col}_scatter.png", fig_dir)

    # === ADVANCED VISUALIZATIONS ===
    logger.info("Generating advanced visualizations...")
    
    # Pass/fail analysis
    outcome_df = df.assign(outcome=lambda d: d["G3"].ge(10).map({True: "Pass", False: "Fail"}))
    
    # Absences by outcome
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=outcome_df, x="outcome", y="absences", ax=ax)
    ax.set(xlabel="Academic Outcome", ylabel="Number of Absences", 
           title="Absences Distribution by Academic Outcome")
    _save_figure(ax, "eda_absences_by_outcome.png", fig_dir)

    # Study time by outcome  
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=outcome_df, x="outcome", y="studytime", ax=ax)
    ax.set(xlabel="Academic Outcome", ylabel="Weekly Study Time", 
           title="Study Time Distribution by Academic Outcome")
    _save_figure(ax, "eda_studytime_by_outcome.png", fig_dir)

    # Grade progression visualization
    if all(col in df.columns for col in ["G1", "G2", "G3"]):
        g = sns.pairplot(df[["G1", "G2", "G3"]], kind="reg", diag_kind="kde", 
                        plot_kws={"scatter_kws": {"s": 30, "alpha": 0.6}})
        g.fig.suptitle("Grade Progression Analysis", y=1.02)
        g.fig.tight_layout()
        g.fig.savefig(fig_dir / "eda_grade_progression_pairplot.png")
        plt.close(g.fig)

    # === GENERATE NARRATIVE SUMMARY ===
    logger.info("Generating narrative summary...")
    summary_text = _generate_eda_summary(df, categorical_importance, categorical_correlations)
    
    # Save narrative summary
    ensure_dir(report_dir)
    with open(report_dir / "eda_comprehensive_report.md", "w") as f:
        f.write(summary_text)
    
    logger.info("EDA complete. Generated %d figures and comprehensive report.", 
                len(list(fig_dir.glob("eda_*.png"))))


__all__ = ["run_eda"]

# Add this execution block
if __name__ == "__main__":
    df = pd.read_csv("student-mat.csv")
    logger.info("Running EDA with student data...")
    run_eda(df)
    logger.info(
        "EDA complete. Check the 'figures' and 'tables' directories for output."
    )



