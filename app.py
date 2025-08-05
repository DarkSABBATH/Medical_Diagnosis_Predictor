import streamlit as st
import joblib
import numpy as np
import os

# --- Generate Placeholder Model (if missing) ---
if not os.path.exists('medical_model.pkl'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Mock data training
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier().fit(X, y)
    scaler = StandardScaler().fit(X)
    
    joblib.dump({
        'diab_model': model,
        'heart_model': model,
        'bc_model': model,
        'park_model': model,
        'scaler': scaler,
        'diabetes_features': ['Glucose','BMI','Age','BP','Insulin'],
        'heart_features': ['Age','BP','Cholesterol','HR','ChestPain'],
        'bc_features': ['Radius','Texture','Perimeter','Area','Smoothness'],
        'park_features': ['Jitter','Shimmer','RPDE','DFA','PPE']
    }, 'medical_model.pkl')

# --- Load Model ---
model_data = joblib.load('medical_model.pkl')

# --- Streamlit UI ---
st.title("Medical Diagnosis Predictor")
disease = st.selectbox("Select Disease", ["diabetes", "heart", "breast_cancer", "parkinsons"])

# Dynamic inputs
inputs = []
for feature in model_data[f"{disease}_features"]:
    inputs.append(st.number_input(feature, value=0.0))

# Prediction
if st.button("Predict Risk"):
    try:
        input_array = np.array(inputs).reshape(1, -1)
        scaled_input = model_data['scaler'].transform(input_array)
        model = model_data[f"{disease}_model"]
        proba = model.predict_proba(scaled_input)[0][1]
        st.success(f"Risk: {proba*100:.1f}%")
    except Exception as e:
        st.error(f"Error: {str(e)}")
