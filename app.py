import streamlit as st
import joblib
import numpy as np

# Load model
model_data = joblib.load('medical_model.pkl')

st.title("Medical Diagnosis Predictor")

disease = st.selectbox("Select Disease", ["diabetes", "heart", "breast_cancer", "parkinsons"])

# Dynamic inputs
inputs = []
for feature in model_data[f'{disease}_features']:
    inputs.append(st.number_input(feature))

if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    scaled_input = model_data['scaler'].transform(input_array)
    proba = model_data[f'{disease}_model'].predict_proba(scaled_input)[0][1]
    st.success(f"Risk: {proba*100:.1f}%")
