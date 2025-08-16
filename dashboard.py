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
import io
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

# --- Absolute imports from your package in src/ ---
# These will work when you run from the project root.
from src.data import load_data
from src.preprocessing import build_pipeline

# from src.model import create_model  # Uncomment if you expose interactive training

# ---------- Configuration ----------
PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"
REPORTS_DIR = PROJECT_DIR / "reports"

# Cache time-to-live in seconds for auto-refreshing cached data
CACHE_TTL = 600

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Student Performance Dashboard")

# ---------- Helpers ----------


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists() and path.is_file():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Could not read CSV: `{path.name}` → {e}")
            return None
    return None



@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def _list_images(
    folder: Path, patterns=(".png", ".jpg", ".jpeg", ".svg", ".webp")
) -> list[Path]:
    if not folder.exists():
        return []
    files: set[Path] = set()
    for ext in patterns:
        # Build a case-insensitive glob pattern such as *.[pP][nN][gG]
        suffix = ext.lstrip(".")
        ci_pattern = "*." + "".join(
            f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in suffix
        )
        files.update(folder.glob(ci_pattern))
    return sorted(files)


@st.cache_data(show_spinner=False)
def _read_file_bytes(path: str) -> bytes:
    """Read file bytes with Streamlit caching."""
    return Path(path).read_bytes()

def _show_images_grid(
    image_paths: list[Path], cols: int = 2, caption_from_name: bool = True, max_per_page: int = 6
):
    if not image_paths:
        st.info("No figures found yet. Generate them via your EDA/training scripts.")
        return
    
    # Add pagination
    total_pages = (len(image_paths) + max_per_page - 1) // max_per_page
    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key=f"page_{hash(str(image_paths))}")
        start_idx = (page - 1) * max_per_page
        end_idx = min(start_idx + max_per_page, len(image_paths))
        current_paths = image_paths[start_idx:end_idx]
        st.caption(f"Showing {start_idx+1}-{end_idx} of {len(image_paths)} images")
    else:
        current_paths = image_paths
        
    # chunk images
    for i in range(0, len(current_paths), cols):
        row_paths = current_paths[i : i + cols]
        cols_objs = st.columns(len(row_paths))
        for c, p in zip(cols_objs, row_paths):
            with c:
                try:
                    img_bytes = _read_file_bytes(str(p))                    
                    if p.suffix.lower() == ".svg":
                        st.image(io.BytesIO(img_bytes))
                    else:
                        st.image(img_bytes)
                    if caption_from_name:
                        st.caption(p.stem.replace("_", " ").title())
                    st.download_button(
                        label="Download",
                        data=img_bytes,
                        file_name=p.name,
                        mime="image/svg+xml" if p.suffix.lower() == ".svg" else None,
                        key=f"dl_{p.name}_{i}",
                    )
                except Exception as e:
                    st.warning(f"Failed to display `{p.name}` → {e}")

def _show_table(csv_path: Path, title: str):
    df = _safe_read_csv(csv_path)
    if df is None:
        st.info(f"`{csv_path.name}` not available. Create it in the `tables/` folder.")
        return
    st.subheader(title)
    st.dataframe(df, use_container_width=True)
    # Offer download
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=csv_path.name,
        mime="text/csv",
        key=f"dl_{csv_path.name}",
    )

# ---------- Page Navigation ----------
tab_names = [
    "EDA Plots",
    "Model Performance",
    "Fairness Metrics",
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
        ("Nested CV Regression Metrics", REPORTS_DIR / "nested_cv_regression_metrics.csv"),
        ("Classification Report",      REPORTS_DIR / "classification_report.csv"),
        ("Best Hyperparameters",       REPORTS_DIR / "best_params.csv"),
    ]
    for title, path in candidate_tables:
        _show_table(path, title)

    st.subheader("Performance Figures")
    perf_imgs = [p for p in _list_images(FIGURES_DIR) if any(
        tag in p.stem.lower() for tag in ("roc", "pr", "confusion", "learning_curve", "residual", "calibration")
    )]
    if perf_imgs:
        _show_images_grid(perf_imgs, cols=2)
    else:
        st.info("No performance figures found. Save ROC/PR/confusion/learning-curve plots into `figures/`.")

elif current_tab == "Fairness Metrics":
    st.header("Fairness Metrics")
    st.markdown(
        "Displays any fairness-related tables or figures you export to **`reports/`** and **`figures/`**. "
        "Run training with `--group-cols` to generate these."
    )
    # Find all fairness reports
    fairness_tables = sorted(REPORTS_DIR.glob("fairness_*.csv"))
    if not fairness_tables:
        st.info("No fairness reports found. Run training with a `--group-cols` argument.")
    for path in fairness_tables:
        _show_table(path, f"Fairness Metrics for {path.stem.split('_', 1)[1]}")

    st.subheader("Fairness Figures")
    fairness_imgs = [p for p in _list_images(FIGURES_DIR) if "fair" in p.stem.lower()]
    _show_images_grid(fairness_imgs, cols=2)

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
        feature_map = {p.stem.split("shap_dependence_", 1)[1]: p for p in shap_dep_paths}
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
        lime_choice = st.selectbox("Select LIME explanation", list(lime_map.keys()), key="lime_select")
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
    _safe_read_csv.clear()
    _list_images.clear()
    _read_file_bytes.clear()
    gc.collect()
    st.sidebar.success("Memory cleaned up")

# Add this button to sidebar:
if st.sidebar.button("Clean Memory"):
    cleanup_memory()