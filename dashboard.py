"""
Student Performance Dashboard (Streamlit)

Project layout assumed:
  project_root/
    dashboard.py            <-- this file
    src/
      __init__.py
      data.py
      preprocessing.py
      model.py
    figures/                <-- PNG/SVG files produced by EDA/training
    tables/                 <-- CSV files (metrics, comparisons, etc.)

Run from the project root:
    streamlit run dashboard.py

Optional (Windows/MKL): set OMP threads to avoid KMeans warning/leak
    $env:OMP_NUM_THREADS=2
"""

from contextlib import contextmanager
from pathlib import Path
import os
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
import altair as alt

# --- Absolute imports from your package in src/ ---
# These will work when you run from the project root.
from src.data import load_data
from src.preprocessing import build_pipeline
from src import (
    _safe_read_csv,
    _list_images,
    _show_images_grid,
    _show_table,
    clear_caches,
)

# from src.model import create_model  # Uncomment if you expose interactive training

# ---------- Configuration ----------
PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"
REPORTS_DIR = PROJECT_DIR / "reports"


st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Optional simple authentication: set DASHBOARD_PASSWORD env var
AUTH_TOKEN = os.getenv("DASHBOARD_PASSWORD")
if AUTH_TOKEN:
    entered = st.sidebar.text_input("Access token", type="password")
    if entered != AUTH_TOKEN:
        st.warning("Invalid token" if entered else "Enter access token")
        st.stop()

st.title("Student Performance Dashboard")
st.caption(
    "GUIDE: A Framework for Guiding Unbiased and Interpretable Decisions in "
    "Education with Explainable and Fair Machine Learning"
)

# ---------- Page Navigation ----------
tab_names = [
    "EDA Plots",
    "Model Performance",
    "OULAD Experiments",
    "Fairness Metrics",
    "Uncertainty Analysis",
    "Transfer Experiments",
    "Counterfactuals",
    "Explanations",
    "Concept Explanations",
]

current_tab = st.sidebar.radio("Navigation", tab_names)

# ---------- Pages ----------

if current_tab == "EDA Plots":
    st.header("Exploratory Data Analysis")
    st.markdown(
        "This section displays pre-generated figures from the **`figures/`** directory. "
        "Run your EDA script first (e.g., `python src/eda.py`) to populate the folder."
    )
    # Show all images, or you can filter by prefixes if you follow a naming scheme
    eda_imgs = _list_images(FIGURES_DIR)
    _show_images_grid(eda_imgs, cols=2)

elif current_tab == "Model Performance":
    st.header("Model Performance")
    st.markdown(
        "Metrics and comparisons are read from CSV files in **`tables/`** and **`reports/`**. "
        "Typical files include: `model_performance.csv`, `classification_report.csv`, "
        "`nested_cv_regression_metrics.csv`, and confusion matrices saved as images."
    )

    # Common tables (show any that exist)
    candidate_tables = [
        ("Model Performance Comparison", TABLES_DIR / "model_performance.csv"),
        (
            "Nested CV Regression Metrics",
            TABLES_DIR / "nested_cv_regression_metrics.csv",
        ),
        ("OULAD Regression Metrics", TABLES_DIR / "oulad_regression_metrics.csv"),
        ("Classification Report", REPORTS_DIR / "classification_report.csv"),
        ("Best Hyperparameters", REPORTS_DIR / "best_params.csv"),
        ("Threshold Tuning", TABLES_DIR / "threshold_tuning.csv"),
    ]

    available_tables = {
        title: path for title, path in candidate_tables if path.exists()
    }
    if available_tables:
        table_selection = st.multiselect(
            "Select performance tables to display",
            list(available_tables.keys()),
            default=list(available_tables.keys()),
        )
        for title in table_selection:
            df = _safe_read_csv(available_tables[title])
            if df is not None:
                st.subheader(title)
                metric_selection = st.multiselect(
                    f"Metrics for {title}",
                    list(df.columns),
                    default=list(df.columns),
                    key=f"metric_select_{title}",
                )
                st.dataframe(df[metric_selection], use_container_width=True)
    else:
        st.info("No performance tables found in `tables/` or `reports/`.")

    st.subheader("Performance Figures")
    # Prepare figure options
    figure_options = {}
    roc_csv = TABLES_DIR / "roc_curve.csv"
    pr_csv = TABLES_DIR / "pr_curve.csv"
    roc_img = FIGURES_DIR / "roc_curve.png"
    pr_img = FIGURES_DIR / "pr_curve.png"
    if roc_csv.exists() or roc_img.exists():
        figure_options["ROC Curve"] = (roc_csv, roc_img)
    if pr_csv.exists() or pr_img.exists():
        figure_options["Precision-Recall Curve"] = (pr_csv, pr_img)

    # Include other performance-related images
    other_imgs = [
        p
        for p in _list_images(FIGURES_DIR)
        if any(
            tag in p.stem.lower()
            for tag in ("confusion", "learning_curve", "residual", "calibration")
        )
        and p.name not in {roc_img.name, pr_img.name}
    ]
    for img in other_imgs:
        figure_options[img.stem.replace("_", " ").title()] = (None, img)

    if figure_options:
        figure_selection = st.multiselect(
            "Select performance figures",
            list(figure_options.keys()),
            default=list(figure_options.keys()),
        )
        for title in figure_selection:
            csv_path, img_path = figure_options[title]
            if csv_path is not None and csv_path.exists():
                df = _safe_read_csv(csv_path)
                if df is not None and not df.empty:
                    if "roc" in title.lower():
                        chart = (
                            alt.Chart(df)
                            .mark_line()
                            .encode(
                                x=alt.X("fpr", title="False Positive Rate"),
                                y=alt.Y("tpr", title="True Positive Rate"),
                            )
                            .interactive()
                        )
                    elif "precision" in title.lower():
                        chart = (
                            alt.Chart(df)
                            .mark_line()
                            .encode(
                                x=alt.X("recall", title="Recall"),
                                y=alt.Y("precision", title="Precision"),
                            )
                            .interactive()
                        )
                    else:
                        chart = (
                            alt.Chart(df)
                            .mark_line()
                            .encode(x=list(df.columns)[0], y=list(df.columns)[1])
                            .interactive()
                        )
                    st.altair_chart(chart, use_container_width=True)
                    continue
            if img_path is not None and img_path.exists():
                st.image(str(img_path), caption=title)
    else:
        st.info(
            "No performance figures found. Save ROC/PR/confusion/learning-curve plots into `figures/` or `tables/`."
        )

elif current_tab == "OULAD Experiments":
    st.header("OULAD Experiments")
    st.markdown(
        "Results for the Open University Learning Analytics Dataset. "
        "Generate outputs with `src/oulad` utilities to populate this section."
    )
    oulad_tables = sorted(TABLES_DIR.glob("oulad_*_metrics.csv"))
    for path in oulad_tables:
        _show_table(path, path.stem.replace("_", " ").title())
    if not oulad_tables:
        st.info("No OULAD tables found. Run OULAD experiments to create them.")
    oulad_imgs = [p for p in _list_images(FIGURES_DIR) if "oulad" in p.stem.lower()]
    _show_images_grid(oulad_imgs, cols=2)

elif current_tab == "Fairness Metrics":
    st.header("Fairness Metrics")
    st.markdown(
        "Displays pre- and post-mitigation fairness reports saved to **`reports/`**. "
        "Run training with `--group-cols` to generate these."
    )
    fairness_pre = sorted(REPORTS_DIR.glob("fairness_*_pre.csv"))
    if not fairness_pre:
        st.info(
            "No fairness reports found. Run training with a `--group-cols` argument."
        )
    for pre_path in fairness_pre:
        base = pre_path.stem.replace("_pre", "")
        post_path = REPORTS_DIR / f"{base}_post.csv"
        pre_df = _safe_read_csv(pre_path)
        post_df = _safe_read_csv(post_path) if post_path.exists() else None
        display_name = base.replace("fairness_", "").replace("_", " ")
        if pre_df is not None and post_df is not None:
            metrics = ["demographic_parity", "equalized_odds"]
            pre_post_cols = [f"{m}_pre" for m in metrics] + [
                f"{m}_post" for m in metrics
            ]
            if all(col in post_df.columns for col in pre_post_cols):
                merged = post_df.copy()
                group_cols = [
                    c
                    for c in merged.columns
                    if c not in pre_post_cols + ["dp_delta", "eo_delta"]
                ]
            else:
                group_cols = [c for c in pre_df.columns if c not in metrics]
                merged = pre_df[group_cols + metrics].merge(
                    post_df[group_cols + metrics],
                    on=group_cols,
                    suffixes=("_pre", "_post"),
                )
            if "dp_delta" not in merged.columns:
                merged["dp_delta"] = (
                    merged["demographic_parity_post"] - merged["demographic_parity_pre"]
                )
            if "eo_delta" not in merged.columns:
                merged["eo_delta"] = (
                    merged["equalized_odds_post"] - merged["equalized_odds_pre"]
                )
            renamed = merged.rename(
                columns={
                    "demographic_parity_pre": "Demographic Parity (Pre)",
                    "demographic_parity_post": "Demographic Parity (Post)",
                    "dp_delta": "DP Δ",
                    "equalized_odds_pre": "Equalized Odds (Pre)",
                    "equalized_odds_post": "Equalized Odds (Post)",
                    "eo_delta": "EO Δ",
                }
            )
            st.subheader(f"Fairness Metrics for {display_name}")
            st.dataframe(
                renamed[
                    group_cols
                    + [
                        "Demographic Parity (Pre)",
                        "Demographic Parity (Post)",
                        "DP Δ",
                        "Equalized Odds (Pre)",
                        "Equalized Odds (Post)",
                        "EO Δ",
                    ]
                ],
                use_container_width=True,
            )

            chart_df = renamed.copy()
            chart_df["group"] = chart_df[group_cols].astype(str).agg("_".join, axis=1)
            dp_chart = chart_df.set_index("group")[
                ["Demographic Parity (Pre)", "Demographic Parity (Post)"]
            ]
            eo_chart = chart_df.set_index("group")[
                ["Equalized Odds (Pre)", "Equalized Odds (Post)"]
            ]
            st.bar_chart(dp_chart, use_container_width=True)
            st.bar_chart(eo_chart, use_container_width=True)

            delta_long = chart_df[["group", "DP Δ", "EO Δ"]].melt(
                "group", var_name="Metric", value_name="Delta"
            )
            delta_chart = (
                alt.Chart(delta_long)
                .mark_bar()
                .encode(
                    x=alt.X("group:N", title="Group"),
                    y=alt.Y("Delta:Q", title="Δ (Post - Pre)"),
                    color=alt.condition(
                        alt.datum.Delta >= 0,
                        alt.value("seagreen"),
                        alt.value("indianred"),
                    ),
                    column=alt.Column("Metric:N", title=None),
                )
            )
            st.altair_chart(delta_chart, use_container_width=True)
        elif pre_df is not None:
            st.subheader(f"Fairness Metrics for {display_name} (pre-mitigation)")
            st.dataframe(pre_df, use_container_width=True)

    st.subheader("Fairness Figures")
    fairness_imgs = [p for p in _list_images(FIGURES_DIR) if "fair" in p.stem.lower()]
    _show_images_grid(fairness_imgs, cols=2)

elif current_tab == "Uncertainty Analysis":
    st.header("Uncertainty Analysis")
    st.markdown(
        "Displays conformal prediction and other uncertainty estimates. "
        "Run uncertainty scripts to generate these artifacts."
    )
    unc_tables = sorted(TABLES_DIR.glob("conformal_*.csv")) + sorted(
        TABLES_DIR.glob("uncertainty_*.csv")
    )
    for path in unc_tables:
        _show_table(path, path.stem.replace("_", " ").title())
    if not unc_tables:
        st.info(
            "No uncertainty tables found. Run uncertainty experiments to create them."
        )
    unc_imgs = [
        p
        for p in _list_images(FIGURES_DIR)
        if any(tag in p.stem.lower() for tag in ("uncert", "conformal"))
    ]
    _show_images_grid(unc_imgs, cols=2)

elif current_tab == "Transfer Experiments":
    st.header("Transfer Experiments")
    st.markdown(
        "Results from cross-dataset transfer learning experiments."
        "Use scripts in `src/transfer` to produce these outputs."
    )
    transfer_tables = sorted(TABLES_DIR.glob("transfer_*.csv"))
    for path in transfer_tables:
        _show_table(path, path.stem.replace("_", " ").title())
    if not transfer_tables:
        st.info(
            "No transfer experiment tables found. Run transfer scripts to create them."
        )
    transfer_imgs = [
        p for p in _list_images(FIGURES_DIR) if "transfer" in p.stem.lower()
    ]
    _show_images_grid(transfer_imgs, cols=2)

elif current_tab == "Counterfactuals":
    st.header("Counterfactual Examples")
    st.markdown(
        "Displays counterfactual tables generated during training. "
        "Files are read from the **`reports/`** directory."
    )
    cf_tables = sorted(REPORTS_DIR.glob("counterfactual_*.csv"))
    cf_tables += sorted(REPORTS_DIR.glob("early_counterfactual_*.csv"))
    if not cf_tables:
        st.info(
            "No counterfactual tables found. Run training scripts to generate them."
        )
    for path in cf_tables:
        _show_table(path, path.stem.replace("_", " ").title())

elif current_tab == "Explanations":
    st.header("Model Explanations")
    st.markdown(
        "Explore global and local explanations generated with **SHAP** and **LIME**. "
        "Save plots in `figures/` and HTML files in `figures/` or `reports/` to view them here."
    )

    shap_summary = FIGURES_DIR / "shap_summary.png"
    if shap_summary.exists():
        st.subheader("SHAP Summary Plot")
        st.image(str(shap_summary))
    else:
        st.info("`shap_summary.png` not found in `figures/`.")

    shap_dep_paths = sorted(FIGURES_DIR.glob("shap_dependence_*.png"))
    if shap_dep_paths:
        st.subheader("SHAP Dependence Plots")
        feature_map = {
            p.stem.split("shap_dependence_", 1)[1]: p for p in shap_dep_paths
        }
        feature = st.selectbox("Select feature", list(feature_map.keys()))
        st.image(str(feature_map[feature]))
    else:
        st.info("No SHAP dependence plots available.")

    lime_paths: list[Path] = []
    lime_paths.extend(FIGURES_DIR.glob("lime_*.html"))
    if REPORTS_DIR.exists():
        lime_paths.extend(REPORTS_DIR.glob("lime_*.html"))
    if lime_paths:
        st.subheader("LIME Explanations")
        lime_map = {p.stem: p for p in sorted(lime_paths)}
        lime_choice = st.selectbox(
            "Select LIME explanation", list(lime_map.keys()), key="lime_select"
        )
        st_html(
            lime_map[lime_choice].read_text(encoding="utf-8"),
            height=600,
            scrolling=True,
        )

    else:
        st.info("No LIME HTML explanations found.")

    shap_html = FIGURES_DIR / "shap_summary.html"
    if shap_html.exists():
        st.subheader("Interactive SHAP Summary")
        try:
            from streamlit_shap import st_shap

            st_shap(open(shap_html).read(), height=600)
        except Exception:
            st_html(shap_html.read_text(), height=600)

    st.subheader("Custom Student Record")
    uploaded = st.file_uploader("Upload a student record (CSV)", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not read uploaded file: {e}")

elif current_tab == "Concept Explanations":
    st.header("Concept-Level Explanations")
    st.markdown(
        "Causal effects of pedagogical concepts on final grades. "
        "Tables and figures are loaded from the `tables/` and `figures/` folders."
    )
    _show_table(TABLES_DIR / "concept_importance.csv", "Concept Importance")
    concept_fig = FIGURES_DIR / "concept_importance.png"
    if concept_fig.exists():
        st.image(str(concept_fig))
    else:
        st.info(
            "`concept_importance.png` not found. Run `python src/concepts.py` to generate it."
        )
# ---------- Optional: lightweight data/pipeline showcase ----------
with st.expander("Quick Sanity Check (loads a few rows)"):
    st.write(
        "This optional check loads the dataset and builds the preprocessing pipeline "
        "to confirm your environment is wired correctly."
    )
    try:
        # Adjust default path if your CSV lives elsewhere
        csv_default = str((PROJECT_DIR / "student-mat.csv").resolve())
        csv_path = st.text_input("CSV path", value=csv_default)
        if st.button("Load sample & build pipeline"):
            X, y = load_data(csv_path)
            st.write("Shape:", X.shape, "Target length:", len(y))
            st.dataframe(X.head(5), use_container_width=True)
            pipe = build_pipeline(X.head(100))  # small sample to avoid heavy work
            st.success("Pipeline created successfully.")
    except Exception as e:
        st.warning(f"Sanity check failed: {e}")


@contextmanager
def st_progress_operation(message="Processing..."):
    """Context manager to show progress during potentially slow operations."""
    progress = st.progress(0)
    status = st.empty()
    status.text(message)
    try:
        yield
    finally:
        progress.progress(100)
        status.empty()


# Add a memory cleanup function:
def cleanup_memory():
    """Force garbage collection and clear caches."""
    import gc

    clear_caches()
    gc.collect()
    st.sidebar.success("Memory cleaned up")


# Add this button to sidebar:
if st.sidebar.button("Clean Memory"):
    cleanup_memory()
