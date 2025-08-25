"""Teacher-Centric Dashboard (Streamlit)

Run from project root:
    streamlit run dashboard_teacher.py
"""

from pathlib import Path
import os
import altair as alt
import streamlit as st

from src.data import load_data
from src import _list_images, _show_images_grid, _show_table

PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

st.set_page_config(page_title="Teacher Dashboard", layout="wide")

AUTH_TOKEN = os.getenv("DASHBOARD_PASSWORD")
if AUTH_TOKEN:
    entered = st.sidebar.text_input("Access token", type="password")
    if entered != AUTH_TOKEN:
        st.warning("Invalid token" if entered else "Enter access token")
        st.stop()

st.title("Teacher Dashboard")

csv_default = str((PROJECT_DIR / "student-mat.csv").resolve())
csv_path = st.sidebar.text_input("Class data CSV", value=csv_default)
try:
    X, y = load_data(csv_path)
    data = X.copy()
    data["pass"] = y
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

overview_tab, risk_tab, fairness_tab = st.tabs(
    [
        "Class Overview",
        "At-Risk Students",
        "Fairness by Subgroup",
    ]
)

with overview_tab:
    st.subheader("Summary")
    st.metric("Number of students", len(data))
    st.metric("Pass rate", f"{data['pass'].mean() * 100:.1f}%")
    st.dataframe(data.head(), use_container_width=True)
    overview_imgs = [
        p for p in _list_images(FIGURES_DIR) if "overview" in p.stem.lower()
    ]
    _show_images_grid(overview_imgs, cols=2)
    overview_tables = sorted(TABLES_DIR.glob("overview_*.csv"))
    for path in overview_tables:
        _show_table(path, path.stem.replace("_", " ").title())

with risk_tab:
    st.subheader("Students At Risk")
    at_risk = data[data["pass"] == 0]
    if at_risk.empty:
        st.success("No at-risk students found.")
    else:
        st.dataframe(at_risk, use_container_width=True)
    risk_tables = sorted(TABLES_DIR.glob("risk_*.csv"))
    for path in risk_tables:
        _show_table(path, path.stem.replace("_", " ").title())

with fairness_tab:
    st.subheader("Fairness by Subgroup")
    cat_cols = [c for c in data.columns if data[c].dtype == "object"]
    if not cat_cols:
        st.info("No categorical columns available.")
    else:
        group_col = st.selectbox("Group column", cat_cols)
        group_df = data.groupby(group_col)["pass"].mean().reset_index()
        group_df["pass_rate"] = group_df["pass"] * 100
        st.dataframe(group_df[[group_col, "pass_rate"]], use_container_width=True)
        chart = (
            alt.Chart(group_df)
            .mark_bar()
            .encode(
                x=alt.X(group_col, title=group_col.title()),
                y=alt.Y("pass_rate", title="Pass Rate (%)"),
            )
        )
        st.altair_chart(chart, use_container_width=True)
    fairness_tables = sorted(TABLES_DIR.glob("fairness_*_pre.csv"))
    for path in fairness_tables:
        _show_table(path, path.stem.replace("_", " ").title())
    fairness_imgs = [p for p in _list_images(FIGURES_DIR) if "fair" in p.stem.lower()]
    _show_images_grid(fairness_imgs, cols=2)
