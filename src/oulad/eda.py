"""
OULAD Dataset Exploratory Data Analysis

Comprehensive EDA for the Open University Learning Analytics Dataset (OULAD).
Generates visualizations and statistical analyses specific to OULAD's structure
including VLE interactions, assessment patterns, and fairness considerations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from logging_config import setup_logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def _save_figure(fig_or_ax, filename: str, fig_dir: Path) -> None:
    """Save figure to file."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(fig_or_ax, 'get_figure'):
        fig = fig_or_ax.get_figure()
    else:
        fig = fig_or_ax
    
    fig.savefig(fig_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved figure: {filename}")


def _save_table(df: pd.DataFrame, filename: str, table_dir: Path) -> None:
    """Save table to CSV file."""
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / filename, index=False)
    logger.info(f"Saved table: {filename}")


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramér's V statistic for categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))


def analyze_student_demographics(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    """Analyze student demographic patterns in OULAD."""
    logger.info("Analyzing student demographics...")
    
    # Demographic distribution
    demographic_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(demographic_cols):
        if col in df.columns:
            df[col].value_counts().plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Remove empty subplot
    if len(demographic_cols) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    _save_figure(fig, "oulad_demographics_distribution.png", fig_dir)
    
    # Demographic summary table
    demo_summary = []
    for col in demographic_cols:
        if col in df.columns:
            counts = df[col].value_counts()
            for category, count in counts.items():
                demo_summary.append({
                    'attribute': col,
                    'category': category,
                    'count': count,
                    'percentage': count / len(df) * 100,
                    'missing': df[col].isna().sum()
                })
    
    demo_df = pd.DataFrame(demo_summary)
    _save_table(demo_df, "oulad_demographics_summary.csv", table_dir)


def analyze_vle_patterns(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    """Analyze VLE interaction patterns."""
    logger.info("Analyzing VLE interaction patterns...")
    
    vle_cols = [col for col in df.columns if 'vle_' in col]
    
    if not vle_cols:
        logger.warning("No VLE columns found")
        return
    
    # VLE activity distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total clicks distribution
    if 'vle_total_clicks' in df.columns:
        axes[0, 0].hist(df['vle_total_clicks'].dropna(), bins=50, alpha=0.7)
        axes[0, 0].set_title('Distribution of Total VLE Clicks')
        axes[0, 0].set_xlabel('Total Clicks')
        axes[0, 0].set_ylabel('Frequency')
    
    # Days active distribution
    if 'vle_days_active' in df.columns:
        axes[0, 1].hist(df['vle_days_active'].dropna(), bins=30, alpha=0.7)
        axes[0, 1].set_title('Distribution of VLE Days Active')
        axes[0, 1].set_xlabel('Days Active')
        axes[0, 1].set_ylabel('Frequency')
    
    # VLE clicks vs outcome
    if 'vle_total_clicks' in df.columns and 'label_pass' in df.columns:
        df.boxplot(column='vle_total_clicks', by='label_pass', ax=axes[1, 0])
        axes[1, 0].set_title('VLE Total Clicks by Pass/Fail')
        axes[1, 0].set_xlabel('Pass (1) / Fail (0)')
    
    # Early vs late VLE activity
    if 'vle_first4_clicks' in df.columns and 'vle_last4_clicks' in df.columns:
        axes[1, 1].scatter(df['vle_first4_clicks'], df['vle_last4_clicks'], alpha=0.5)
        axes[1, 1].set_title('Early vs Late VLE Activity')
        axes[1, 1].set_xlabel('First 4 Weeks Clicks')
        axes[1, 1].set_ylabel('Last 4 Weeks Clicks')
    
    plt.tight_layout()
    _save_figure(fig, "oulad_vle_patterns.png", fig_dir)
    
    # VLE summary statistics
    vle_summary = df[vle_cols].describe()
    _save_table(vle_summary, "oulad_vle_summary_stats.csv", table_dir)


def analyze_assessment_patterns(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    """Analyze assessment submission and performance patterns."""
    logger.info("Analyzing assessment patterns...")
    
    assessment_cols = [col for col in df.columns if 'assessment_' in col]
    
    if not assessment_cols:
        logger.warning("No assessment columns found")
        return
    
    # Assessment performance distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Assessment count distribution
    if 'assessment_count' in df.columns:
        axes[0, 0].hist(df['assessment_count'].dropna(), bins=20, alpha=0.7)
        axes[0, 0].set_title('Distribution of Assessment Count')
        axes[0, 0].set_xlabel('Number of Assessments')
        axes[0, 0].set_ylabel('Frequency')
    
    # Mean score distribution
    if 'assessment_mean_score' in df.columns:
        axes[0, 1].hist(df['assessment_mean_score'].dropna(), bins=30, alpha=0.7)
        axes[0, 1].set_title('Distribution of Mean Assessment Score')
        axes[0, 1].set_xlabel('Mean Score')
        axes[0, 1].set_ylabel('Frequency')
    
    # Assessment performance vs outcome
    if 'assessment_mean_score' in df.columns and 'label_pass' in df.columns:
        df.boxplot(column='assessment_mean_score', by='label_pass', ax=axes[1, 0])
        axes[1, 0].set_title('Assessment Mean Score by Pass/Fail')
        axes[1, 0].set_xlabel('Pass (1) / Fail (0)')
    
    # On-time submission rate
    if 'assessment_ontime_rate' in df.columns:
        axes[1, 1].hist(df['assessment_ontime_rate'].dropna(), bins=20, alpha=0.7)
        axes[1, 1].set_title('Distribution of On-time Submission Rate')
        axes[1, 1].set_xlabel('On-time Rate')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    _save_figure(fig, "oulad_assessment_patterns.png", fig_dir)
    
    # Assessment summary statistics
    assessment_summary = df[assessment_cols].describe()
    _save_table(assessment_summary, "oulad_assessment_summary_stats.csv", table_dir)


def analyze_fairness_patterns(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    """Analyze fairness and bias patterns across sensitive attributes."""
    logger.info("Analyzing fairness patterns...")
    
    sensitive_attrs = ['gender', 'age_band', 'highest_education', 'imd_band']
    
    if 'label_pass' not in df.columns:
        logger.warning("No target label found for fairness analysis")
        return
    
    # Pass rates by sensitive attributes
    fairness_summary = []
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, attr in enumerate(sensitive_attrs):
        if attr in df.columns and i < len(axes):
            # Calculate pass rates
            pass_rates = df.groupby(attr)['label_pass'].agg(['mean', 'count'])
            pass_rates.columns = ['pass_rate', 'count']
            
            # Visualization
            pass_rates['pass_rate'].plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Pass Rate by {attr.replace("_", " ").title()}')
            axes[i].set_ylabel('Pass Rate')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Summary for table
            for category in pass_rates.index:
                fairness_summary.append({
                    'sensitive_attribute': attr,
                    'category': category,
                    'pass_rate': pass_rates.loc[category, 'pass_rate'],
                    'count': pass_rates.loc[category, 'count'],
                    'pass_count': int(pass_rates.loc[category, 'pass_rate'] * pass_rates.loc[category, 'count'])
                })
    
    plt.tight_layout()
    _save_figure(fig, "oulad_fairness_pass_rates.png", fig_dir)
    
    # Save fairness summary
    fairness_df = pd.DataFrame(fairness_summary)
    _save_table(fairness_df, "oulad_fairness_summary.csv", table_dir)


def analyze_feature_importance(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    """Analyze feature importance for predicting student success."""
    logger.info("Analyzing feature importance...")
    
    if 'label_pass' not in df.columns:
        logger.warning("No target label found for feature importance analysis")
        return
    
    # Prepare features for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'label_pass']
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Mutual information for numeric features
    if numeric_cols:
        X_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df['label_pass'].fillna(0)
        
        mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
        
        # Create feature importance plot
        importance_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mutual Information Score')
        plt.title('Top 15 Feature Importance for Student Success Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        _save_figure(plt.gcf(), "oulad_feature_importance.png", fig_dir)
        
        _save_table(importance_df, "oulad_feature_importance.csv", table_dir)
    
    # Categorical feature analysis
    if categorical_cols:
        cat_importance = []
        
        # Encode categorical variables
        df_encoded = df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        # Calculate mutual information for categorical features
        X_cat = df_encoded[categorical_cols].fillna(0)
        y = df['label_pass'].fillna(0)
        
        mi_scores_cat = mutual_info_classif(X_cat, y, random_state=42)
        
        for i, col in enumerate(categorical_cols):
            cat_importance.append({
                'feature': col,
                'mutual_info': mi_scores_cat[i],
                'unique_values': df[col].nunique()
            })
        
        cat_importance_df = pd.DataFrame(cat_importance).sort_values('mutual_info', ascending=False)
        _save_table(cat_importance_df, "oulad_categorical_importance.csv", table_dir)


def generate_oulad_summary_report(df: pd.DataFrame, report_dir: Path) -> None:
    """Generate a comprehensive summary report of OULAD EDA findings."""
    logger.info("Generating summary report...")
    
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report = f"""# OULAD Dataset Exploratory Data Analysis Report

## Dataset Overview
- **Total Students**: {len(df):,}
- **Total Features**: {len(df.columns)}
- **Pass Rate**: {df['label_pass'].mean():.2%} (if label_pass exists)
- **Missing Data**: {df.isnull().sum().sum():,} values across all features

## Key Findings

### Demographics
- **Gender Distribution**: {df['gender'].value_counts().to_dict() if 'gender' in df.columns else 'N/A'}
- **Age Distribution**: {df['age_band'].value_counts().to_dict() if 'age_band' in df.columns else 'N/A'}

### VLE Engagement
- **Average Total Clicks**: {df['vle_total_clicks'].mean():.0f} (if available)
- **Average Days Active**: {df['vle_days_active'].mean():.1f} (if available)

### Assessment Performance
- **Average Assessment Count**: {df['assessment_count'].mean():.1f} (if available)
- **Average Score**: {df['assessment_mean_score'].mean():.1f} (if available)

### Fairness Considerations
Detailed fairness analysis across sensitive attributes is available in the fairness summary tables.

## Recommendations
1. Focus on early VLE engagement patterns for intervention
2. Monitor assessment submission patterns for at-risk identification
3. Consider fairness implications across demographic groups
4. Use top predictive features for model development

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_path = report_dir / "oulad_eda_summary.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Summary report saved to {report_path}")


def run_oulad_eda(
    input_path: str = "data/oulad/processed/oulad_ml.parquet",
    fig_dir: str = "figures/oulad",
    table_dir: str = "tables/oulad", 
    report_dir: str = "reports/oulad"
) -> None:
    """
    Run comprehensive EDA on OULAD dataset.
    
    Args:
        input_path: Path to processed OULAD dataset
        fig_dir: Directory to save figures
        table_dir: Directory to save tables
        report_dir: Directory to save reports
    """
    logger.info("Starting OULAD EDA...")
    
    # Convert to Path objects
    fig_dir = Path(fig_dir)
    table_dir = Path(table_dir)
    report_dir = Path(report_dir)
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Run analysis modules
    analyze_student_demographics(df, fig_dir, table_dir)
    analyze_vle_patterns(df, fig_dir, table_dir)
    analyze_assessment_patterns(df, fig_dir, table_dir)
    analyze_fairness_patterns(df, fig_dir, table_dir)
    analyze_feature_importance(df, fig_dir, table_dir)
    
    # Generate summary report
    generate_oulad_summary_report(df, report_dir)
    
    logger.info("OULAD EDA completed successfully!")
    logger.info(f"Outputs saved to:")
    logger.info(f"  Figures: {fig_dir}")
    logger.info(f"  Tables: {table_dir}")
    logger.info(f"  Reports: {report_dir}")


if __name__ == "__main__":
    setup_logging()
    run_oulad_eda()