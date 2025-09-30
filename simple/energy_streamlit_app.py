import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title="Smart City Energy Prediction", page_icon="⚡", layout="centered")

st.markdown("""
    <style>
    .main-title {font-size:2.5rem; color:#1f77b4; text-align:center; margin-bottom:1rem;}
    .result-high {background:#ffeaea; color:#c62828; font-size:1.5rem; padding:1rem; border-radius:0.5rem;}
    .result-low {background:#eaffea; color:#2e7d32; font-size:1.5rem; padding:1rem; border-radius:0.5rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ Smart City Energy Prediction</div>', unsafe_allow_html=True)

models_dir = 'saved_models'
config_path = os.path.join(models_dir, 'model_config.json')
scaler_path = os.path.join(models_dir, 'feature_scaler.pkl')

# Load config and models
def load_config_and_models():
    with open(config_path, 'r') as f:
        config = json.load(f)
    scaler = joblib.load(scaler_path)
    models = {}
    for name in config['model_performances'].keys():
        model_path = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.pkl")
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
    return config, scaler, models

config, scaler, models = load_config_and_models()

st.write(f"**Target Variable:** {config['target_variable']}")
st.write(f"**Best Model:** {config['best_model']}")
st.write(f"**Features Used:** {len(config['selected_features'])}")

st.markdown("---")
st.header("🔮 Make a Prediction")

# Input fields for ALL selected features (to match scaler requirements)
st.subheader("Enter Feature Values:")

# Add sample data button
if st.button("📋 Load Sample Data", help="Load realistic sample values"):
    try:
        # Try to load original dataset for sample values
        df_sample = pd.read_csv('data/smart_city_energy_dataset.csv')
        sample_row = df_sample.sample(1).iloc[0]
        st.session_state.sample_loaded = True
        for feature in config['selected_features']:
            if feature in df_sample.columns:
                st.session_state[f"sample_{feature}"] = float(sample_row[feature])
    except:
        st.info("Sample data not available, using default values")

col1, col2 = st.columns(2)

inputs = {}
for i, feature in enumerate(config['selected_features']):
    # Use sample data if loaded
    default_val = 0.0
    if hasattr(st.session_state, 'sample_loaded') and f"sample_{feature}" in st.session_state:
        default_val = st.session_state[f"sample_{feature}"]
    
    if i % 2 == 0:
        with col1:
            inputs[feature] = st.number_input(f"{feature}", value=default_val, key=f"feat_{i}")
    else:
        with col2:
            inputs[feature] = st.number_input(f"{feature}", value=default_val, key=f"feat_{i}")

if st.button("🚀 Predict with All Models", type="primary"):
    # Create input DataFrame with all required features
    input_df = pd.DataFrame([inputs])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    st.subheader("🏆 Predictions from All Models:")
    
    # Create columns for displaying results
    cols = st.columns(len(models))
    
    for i, (name, model) in enumerate(models.items()):
        with cols[i]:
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(input_scaled)[0][1]
                confidence_text = f"Confidence: {prob:.1%}"
            else:
                confidence_text = "SVM (no probability)"
            
            # Display result
            st.markdown(f"**{name}**")
            if prediction == 1:
                st.error("🔴 HIGH")
            else:
                st.success("🟢 LOW")
            st.caption(confidence_text)
            
            # Show performance score
            if name in config['model_performances']:
                perf = config['model_performances'][name]
                st.caption(f"Model F1: {perf:.3f}")
    
    # Show best model highlight
    st.markdown("---")
    best_model = config['best_model']
    if best_model in models:
        best_prediction = models[best_model].predict(input_scaled)[0]
        result_text = "HIGH ENERGY CONSUMPTION" if best_prediction == 1 else "LOW ENERGY CONSUMPTION"
        result_color = "🔴" if best_prediction == 1 else "🟢"
        
        st.success(f"🏆 **Best Model ({best_model}) Prediction:** {result_color} {result_text}")
        
        if hasattr(models[best_model], 'predict_proba'):
            best_prob = models[best_model].predict_proba(input_scaled)[0][1]
            st.info(f"📊 **Confidence Level:** {best_prob:.1%}")

st.markdown("---")
st.caption("Made with Streamlit • Smart City Energy Project • Modern UI")
