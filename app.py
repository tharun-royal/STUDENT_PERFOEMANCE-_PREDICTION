from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_score_model.joblib"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Prediction")
st.write(
    "Enter study details to estimate a student's final score. "
    "This is a classroom demonstration using an illustrative sample dataset, "
    "not a validated tool for grading real students."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file is missing. Run `python train_model.py` first.")
        st.stop()
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]

model, features = load_model()

with st.form("student_form"):
    hours = st.slider("Hours studied per day", 0.0, 12.0, 3.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 80)
    previous = st.slider("Previous exam score", 0, 100, 65)
    assignments = st.slider("Assignments completed (out of 10)", 0, 10, 7)
    submitted = st.form_submit_button("Predict final score")

if submitted:
    sample = pd.DataFrame([{
        "hours_studied": hours,
        "attendance_percent": attendance,
        "previous_score": previous,
        "assignments_completed": assignments,
    }], columns=features)
    prediction = float(model.predict(sample)[0])
    prediction = max(0.0, min(100.0, prediction))
    st.metric("Estimated final score", f"{prediction:.1f} / 100")
    if prediction >= 75:
        st.info("Model estimate: higher score range.")
    elif prediction >= 50:
        st.info("Model estimate: middle score range.")
    else:
        st.warning("Model estimate: lower score range.")
    st.caption("The estimate may be inaccurate. Do not use it to make high-stakes decisions.")

with st.expander("About this demo"):
    st.write("""
    **Inputs:** study hours, attendance, previous score, assignments completed.

    **Target:** final exam score (0–100).

    **Algorithm:** Linear Regression, a supervised learning regression algorithm.

    **Dataset:** generated sample data for teaching. It is not actual student records.
    """)
