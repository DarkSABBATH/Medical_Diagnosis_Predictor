import streamlit as st
import joblib
import numpy as np

# Load model
def load_model():
    return joblib.load('medical_model.pkl')

st.title("Medical Diagnosis Predictor")

model_data = load_model()
disease = st.selectbox("Select Disease", 
                      ["diabetes", "heart", "breast_cancer", "parkinsons"])

# Dynamic inputs
inputs = []
for feature in model_data[f'{disease}_features']:
    inputs.append(st.number_input(feature, value=0.0))

if st.button("Predict"):
    try:
        input_array = np.array(inputs).reshape(1, -1)
        scaled_input = model_data['scaler'].transform(input_array)
        proba = model_data[f'{disease}_model'].predict_proba(scaled_input)[0][1]
        st.success(f"Risk: {proba*100:.1f}%")
        
        # Show top 3 important features
        importances = model_data[f'{disease}_model'].feature_importances_
        top_features = sorted(zip(model_data[f'{disease}_features'], importances),
                            key=lambda x: x[1], reverse=True)[:3]
        st.write("Key factors:", [f[0] for f in top_features])
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
