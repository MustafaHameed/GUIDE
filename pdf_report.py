# pdf_report.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

root = Path(__file__).resolve().parent
fig_dir = root / "figures"
tab_dir = root / "tables"

tables = [
    ("Correlation Matrix", tab_dir / "correlation_matrix.csv",
     "Measures pairwise Pearson correlations between numeric features and final grade."),
    ("Grade by Sex", tab_dir / "grade_by_sex.csv",
     "Shows the distribution of final grades split by student sex."),
    ("Grade by Studytime", tab_dir / "grade_by_studytime.csv",
     "Compares grade outcomes across study‑time categories."),
    ("Nested CV Metrics", tab_dir / "nested_cv_metrics.csv",
     "Cross‑validated performance metrics of the predictive model."),
    ("Statistical Tests", tab_dir / "statistical_tests.csv",
     "P‑values and statistics from hypothesis tests on key features."),
    ("Summary", tab_dir / "summary.csv",
     "Descriptive statistics of the cleaned dataset.")
]

figures = [
    ("G3 Distribution", fig_dir / "g3_distribution.png",
     "Histogram of final‑grade values."),
    ("Grades Distribution", fig_dir / "grades_distribution.png",
     "Pairwise scatter plots between G1, G2, and G3."),
    ("Correlation Heatmap", fig_dir / "correlation_heatmap.png",
     "Correlation matrix visualized as a heatmap."),
    ("Grades Pairplot", fig_dir / "grades_pairplot.png",
     "Pairplot showing relationships between grades at different stages."),
    ("G3 by Sex", fig_dir / "g3_by_sex.png",
     "Boxplot of final grades grouped by sex."),
    ("Studytime vs G3", fig_dir / "studytime_vs_g3.png",
     "Scatter plot of study time against final grade."),
    ("Absences vs G3", fig_dir / "absences_vs_g3.png",
     "Relationship between class absences and final grade."),
    ("Predicted vs Actual", fig_dir / "pred_vs_actual.png",
     "Model predictions compared with actual grades."),
    ("Residuals by School", fig_dir / "residuals_by_school.png",
     "Residual plot split by school."),
    ("Residuals by Sex", fig_dir / "residuals_by_sex.png",
     "Residual plot split by sex."),
    ("Learning Curve", fig_dir / "learning_curve.png",
     "Training/validation score vs. training set size."),
    ("SHAP Summary", fig_dir / "shap_summary.png",
     "SHAP value summary highlighting feature importance."),
    ("SHAP Dependence – G1", fig_dir / "shap_dependence_num__G1.png",
     "SHAP dependence plot for feature G1."),
    ("SHAP Dependence – G2", fig_dir / "shap_dependence_num__G2.png",
     "SHAP dependence plot for feature G2."),
    ("SHAP Dependence – Absences", fig_dir / "shap_dependence_num__absences.png",
     "SHAP dependence plot for absences."),
    ("SHAP Dependence – Age", fig_dir / "shap_dependence_num__age.png",
     "SHAP dependence plot for age."),
    ("SHAP Dependence – FamRel", fig_dir / "shap_dependence_num__famrel.png",
     "SHAP dependence plot for family relationship quality.")
]

with PdfPages("full_report.pdf") as pdf:
    # Format tables
    for title, path, interpretation in tables:
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")
        ax.table(cellText=df.values, colLabels=df.columns,
                 loc="center", cellLoc="center")
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)
        fig.text(0.5, 0.05, interpretation, ha="center", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Insert figures
    for title, path, interpretation in figures:
        img = plt.imread(path)
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.imshow(img)
        ax.axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)
        fig.text(0.5, 0.05, interpretation, ha="center", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
