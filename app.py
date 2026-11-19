import streamlit as st
import joblib
import numpy as np

model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("diabees_scaler.joblib")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0f172a; }
[data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
h1, p, label, button { color: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)

st.title("Diabetes Risk Analyzer")

features = [
    'Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 'HbA1c', 
    'LDL', 'HDL', 'Triglycerides', 'WaistCircumference', 'HipCircumference', 
    'WHR', 'FamilyHistory', 'DietType', 'Hypertension', 'MedicationUse'
]

cols = st.columns(2)
inputs = [cols[i % 2].number_input(f, value=0.0) for i, f in enumerate(features)]

if st.button("Analyze Patient", type="primary"):
    scaled_data = scaler.transform(np.array([inputs]))
    prediction = model.predict(scaled_data)[0]
    
    if prediction == 1:
        st.error("High risk of diabetes detected")
    else:
        st.success("Low risk of diabetes detected")
