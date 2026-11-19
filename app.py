import streamlit as st
import joblib
import numpy as np

model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("diabees_scaler.joblib")

st.title("Diabetes Prediction App")

features = [
    'Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 'HbA1c', 
    'LDL', 'HDL', 'Triglycerides', 'WaistCircumference', 'HipCircumference', 
    'WHR', 'FamilyHistory', 'DietType', 'Hypertension', 'MedicationUse'
]

inputs = [st.number_input(f"Enter {feature}", value=0.0) for feature in features]

if st.button("Predict"):
    scaled_data = scaler.transform(np.array([inputs]))
    prediction = model.predict(scaled_data)[0]
    
    if prediction == 1:
        st.error("There is chance of having diabetes")
    else:
        st.success("There is no chance of having diabetes")