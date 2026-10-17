import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Prediction")
st.write("Estimate an employee's salary using experience, education level, and job role.")
st.caption("Educational demo only: results depend on the small synthetic dataset included with this project.")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model.pkl. Please run the training script first. Details: {exc}")
    st.stop()

with st.form("salary_form"):
    experience = st.number_input("Experience (years)", min_value=0.0, max_value=60.0, value=2.0, step=0.5)
    education = st.selectbox("Education level", ["Degree", "BTech", "MTech"])
    role = st.selectbox("Job role", ["Developer", "Tester", "Analyst"])
    submitted = st.form_submit_button("Predict Salary")

if submitted:
    sample = pd.DataFrame([{
        "experience_years": experience,
        "education_level": education,
        "job_role": role
    }])
    prediction = float(model.predict(sample)[0])
    st.success(f"Predicted salary: ₹{prediction:,.0f} per year")
    st.info("This is an estimate from demo data, not a guaranteed or market-validated salary.")
