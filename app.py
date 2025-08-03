import gradio as gr
import joblib
import numpy as np

# Load trained models
model_data = joblib.load('medical_model.pkl')

def predict(disease, *inputs):
    try:
        # Get model and feature names
        model = model_data[f'{disease}_model']
        features = model_data[f'{disease}_features']
        
        # Convert inputs to array and scale
        input_array = np.array([float(x) for x in inputs]).reshape(1, -1)
        scaled_input = model_data['scaler'].transform(input_array)
        
        # Predict
        proba = model.predict_proba(scaled_input)[0][1]
        
        # Get top 3 important features
        importances = model.feature_importances_
        top_features = sorted(zip(features, importances), 
                            key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "risk": f"{proba*100:.1f}%",
            "top_factors": [f[0] for f in top_features]
        }
    except Exception as e:
        return {"error": str(e)}

# Create interface
with gr.Blocks(title="Medical Diagnosis Predictor") as app:
    gr.Markdown("## 🏥 Predict Disease Risk")
    
    with gr.Row():
        disease = gr.Dropdown(
            choices=["diabetes", "heart", "breast_cancer", "parkinsons"],
            label="Select Disease"
        )
    
    # Dynamic input fields
    inputs = []
    for i in range(10):  # Adjust based on max features needed
        inputs.append(gr.Number(label=f"Feature {i+1}", visible=False))
    
    # Show/hide inputs based on disease selection
    def update_inputs(disease):
        num_features = {
            "diabetes": 8,
            "heart": 13,
            "breast_cancer": 30,
            "parkinsons": 22
        }
        return [
            gr.Number(visible=i < num_features[disease]) 
            for i in range(10)
        ]
    
    disease.change(update_inputs, disease, inputs)
    
    submit = gr.Button("Predict Risk", variant="primary")
    output = gr.JSON()
    
    submit.click(
        predict,
        inputs=[disease] + inputs,
        outputs=output
    )

app.launch()