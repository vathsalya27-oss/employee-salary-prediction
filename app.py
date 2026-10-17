import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Prediction")
st.write("Estimate an employee's salary using experience, education level, and job role.")
st.caption("Educational demo only: results depend on the small synthetic dataset included with this project.")

MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    data = {
        'Experience': [1,2,3,4,5,6,7,8,9,10,12,15,3,5,8],
        'Education': ['Bachelors','Masters','Bachelors','Masters','PhD','Bachelors','Masters','PhD','Bachelors','Masters','PhD','Masters','PhD','Bachelors','Masters'],
        'JobRole': ['Developer','Developer','Analyst','Manager','Manager','Analyst','Developer','Manager','Analyst','Developer','Director','Director','Analyst','Manager','Analyst'],
        'Salary': [40000,50000,55000,75000,90000,65000,80000,110000,60000,85000,130000,125000,70000,70000,75000]
    }
    df = pd.DataFrame(data)
    le_edu = LabelEncoder()
    le_role = LabelEncoder()
    df['Education_enc'] = le_edu.fit_transform(df['Education'])
    df['JobRole_enc'] = le_role.fit_transform(df['JobRole'])
    X = df[['Experience','Education_enc','JobRole_enc']]
    y = df['Salary']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((model, le_edu, le_role), f)

with open(MODEL_PATH, 'rb') as f:
    model, le_edu, le_role = pickle.load(f)

exp = st.slider("Years of Experience", 0, 20, 3)
edu = st.selectbox("Education Level", list(le_edu.classes_))
role = st.selectbox("Job Role", list(le_role.classes_))

if st.button("Predict Salary", type="primary"):
    edu_enc = le_edu.transform([edu])[0]
    role_enc = le_role.transform([role])[0]
    pred = model.predict([[exp, edu_enc, role_enc]])[0]
    st.success(f"Estimated Salary: ${pred:,.2f}")
    st.balloons()
