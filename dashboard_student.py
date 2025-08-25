"""Student-Centric Dashboard (Streamlit)

Run from project root:
    streamlit run dashboard_student.py
"""

from pathlib import Path
import os
import pandas as pd
import altair as alt
import streamlit as st

from src.data import load_data
from src.preprocessing import build_pipeline
from src import _safe_read_csv, clear_caches

PROJECT_DIR = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_DIR / "tables"

st.set_page_config(page_title="Student Dashboard", layout="wide")

# -------- Authentication --------
# Environment variable format: "id1:token1,id2:token2"
_token_map_env = os.getenv("STUDENT_TOKENS")
_token_to_id: dict[str, str] = {}
if _token_map_env:
    for pair in _token_map_env.split(","):
        if ":" in pair:
            sid, tok = pair.split(":", 1)
            _token_to_id[tok.strip()] = sid.strip()

if _token_to_id:
    entered = st.sidebar.text_input("Access token", type="password")
    student_id = _token_to_id.get(entered)
    if student_id is None:
        st.warning("Invalid token" if entered else "Enter access token")
        st.stop()
else:
    student_id = None  # No auth configured

st.title("My Learning Dashboard")

# -------- Data Selection --------
source = st.sidebar.radio("Data Source", ["Demo dataset", "Upload CSV"])

if source == "Demo dataset":
    csv_default = str((PROJECT_DIR / "student-mat.csv").resolve())
    csv_path = st.sidebar.text_input("Class data CSV", value=csv_default)
    try:
        raw_df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
    X, y = load_data(csv_path)
    data = raw_df.copy()
    data["pass"] = y
    ids = data.index.astype(str)
    if student_id is None:
        selected_id = st.sidebar.selectbox("Student ID", ids)
    else:
        if student_id not in ids:
            st.error("ID not in dataset")
            st.stop()
        selected_id = student_id
    sid = int(selected_id)
    student_row = data.loc[sid]
    student_features = X.loc[[sid]]
else:
    uploaded = st.sidebar.file_uploader("Upload your CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV to continue")
        st.stop()
    raw_df = pd.read_csv(uploaded)
    ids = raw_df.index.astype(str)
    if student_id is None:
        selected_id = st.sidebar.selectbox("Row", ids)
    else:
        if student_id not in ids:
            st.error("ID not in uploaded file")
            st.stop()
        selected_id = student_id
    sid = int(selected_id)
    student_row = raw_df.loc[sid]
    # Train model on demo data, then apply to uploaded row
    X, y = load_data(PROJECT_DIR / "student-mat.csv")
    student_features = student_row[X.columns].to_frame().T

# -------- Modeling --------
pipe = build_pipeline(X)
pipe.fit(X, y)
prob_pass = pipe.predict_proba(student_features)[0, 1]
risk = 1 - prob_pass
st.metric("Predicted risk of failing", f"{risk * 100:.1f}%")

# -------- Progress Chart --------
grade_cols = [c for c in ["G1", "G2", "G3"] if c in student_row]
if grade_cols:
    progress_df = pd.DataFrame(
        {
            "exam": grade_cols,
            "grade": [student_row[c] for c in grade_cols],
        }
    )
    chart = (
        alt.Chart(progress_df)
        .mark_line(point=True)
        .encode(x=alt.X("exam", title="Exam"), y=alt.Y("grade", title="Grade"))
    )
    st.altair_chart(chart, use_container_width=True)

# -------- Recommended Resources --------
st.subheader("Recommended Resources")
resources_df = _safe_read_csv(TABLES_DIR / "resources.csv")
if resources_df is not None:
    st.dataframe(resources_df, use_container_width=True)
else:
    if risk > 0.5:
        resources = [
            "Schedule time with a tutor",
            "Join a study group",
            "Review class materials twice a week",
        ]
    else:
        resources = [
            "Keep up the good work!",
            "Explore enrichment activities",
        ]
    for r in resources:
        st.write(f"- {r}")

st.button("Clear caches", on_click=clear_caches)
