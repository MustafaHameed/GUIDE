"""Streamlit dashboard for exploring EDA results, model performance,
 and fairness metrics for the Student Performance dataset.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

# Directories containing pre-generated figures and tables
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Student Performance Dashboard")

page = st.sidebar.selectbox(
    "Navigation",
    ["EDA Plots", "Model Performance", "Fairness Metrics"],
)

if page == "EDA Plots":
    st.header("Exploratory Data Analysis")
    eda_images = [
        "g3_distribution.png",
        "grades_distribution.png",
        "correlation_heatmap.png",
        "grades_pairplot.png",
        "absences_vs_g3.png",
        "studytime_vs_g3.png",
        "g3_by_sex.png",
    ]
    for image in eda_images:
        path = FIGURES_DIR / image
        if path.exists():
            st.subheader(path.stem.replace("_", " ").title())
            st.image(str(path))
        else:
            st.info(f"Missing figure: {image}")

elif page == "Model Performance":
    st.header("Model Performance Summary")
    metrics_file = TABLES_DIR / "nested_cv_metrics.csv"
    if metrics_file.exists():
        metrics_df = pd.read_csv(metrics_file)
        summary = (
            metrics_df.groupby("model")[["rmse", "mae", "r2"]]
            .mean()
            .reset_index()
        )
        st.subheader("Cross-Validation Metrics")
        st.dataframe(summary)
    else:
        st.info("Metrics table not found.")

    perf_images = [
        "learning_curve.png",
        "pred_vs_actual.png",
        "residuals_by_school.png",
        "residuals_by_sex.png",
        "shap_summary.png",
    ]
    for image in perf_images:
        path = FIGURES_DIR / image
        if path.exists():
            st.subheader(path.stem.replace("_", " ").title())
            st.image(str(path))
        else:
            st.info(f"Missing figure: {image}")

elif page == "Fairness Metrics":
    st.header("Per-Group Fairness Metrics")
    fairness_tables = {
        "Sex": TABLES_DIR / "grade_by_sex.csv",
        "Study Time": TABLES_DIR / "grade_by_studytime.csv",
    }
    for name, path in fairness_tables.items():
        if path.exists():
            df = pd.read_csv(path)
            st.subheader(name)
            st.dataframe(df)
            if "mean" in df.columns:
                st.bar_chart(df.set_index(df.columns[0])["mean"])
        else:
            st.info(f"Missing table: {path.name}")

    fairness_fig = FIGURES_DIR / "g3_by_sex.png"
    if fairness_fig.exists():
        st.subheader("G3 by Sex")
        st.image(str(fairness_fig))
    else:
        st.info("Missing figure: g3_by_sex.png")
