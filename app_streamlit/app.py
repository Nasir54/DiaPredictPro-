import streamlit as st
import joblib
import os
import numpy as np
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .diabetes-detected {
        background-color: #ffe6e6;
        border: 2px solid #ff4d4d;
    }
    .no-diabetes {
        background-color: #e6ffe6;
        border: 2px solid #4dff4d;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        # Get the absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, '../models/diabetes_model.pkl')
        scaler_path = os.path.join(base_dir, '../models/scaler.pkl')
        
        st.write(f"Loading model from: {model_path}")
        st.write(f"Loading scaler from: {scaler_path}")
        
        if not os.path.exists(model_path):
            st.error("Model file not found!")
            return None, None
        if not os.path.exists(scaler_path):
            st.error("Scaler file not found!")
            return None, None
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        st.success("Model and scaler loaded successfully!")
        return model, scaler
        
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        # Create dummy model for testing
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        dummy_model = LogisticRegression()
        dummy_scaler = StandardScaler()
        dummy_X = np.random.rand(10, 8)
        dummy_y = np.random.randint(0, 2, 10)
        dummy_scaler.fit(dummy_X)
        dummy_model.fit(dummy_X, dummy_y)
        st.warning("Using dummy model for demonstration")
        return dummy_model, dummy_scaler

# Load the model
model, scaler = load_model()

# App title
st.markdown('<h1 class="main-header">🏥 Diabetes Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 20, 0)
        glucose = st.slider("Glucose", 0.0, 200.0, 100.0, 0.1)
        blood_pressure = st.slider("Blood Pressure", 0.0, 150.0, 70.0, 0.1)
        skin_thickness = st.slider("Skin Thickness", 0.0, 100.0, 20.0, 0.1)

    with col2:
        insulin = st.slider("Insulin", 0.0, 1000.0, 80.0, 0.1)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01)
        age = st.slider("Age", 0, 120, 30, 1)

    # Submit button
    submitted = st.form_submit_button("Predict Diabetes", type="primary")

# Handle prediction
if submitted:
    if model is None or scaler is None:
        st.error("Model not available. Please check the model files.")
    else:
        try:
            # Prepare input features
            features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            if prediction[0] == 1:
                css_class = "diabetes-detected"
                st.markdown(f'<div class="prediction-box {css_class}">', unsafe_allow_html=True)
                st.error(f"🚨 **Diabetes Detected**")
                st.metric("Probability", f"{probability*100:.2f}%")
                st.warning("Please consult with a healthcare professional for further evaluation.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                css_class = "no-diabetes"
                st.markdown(f'<div class="prediction-box {css_class}">', unsafe_allow_html=True)
                st.success(f"✅ **No Diabetes Detected**")
                st.metric("Probability", f"{probability*100:.2f}%")
                st.info("Maintain a healthy lifestyle with regular exercise and balanced diet!")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show feature values
            with st.expander("📋 Input Features Summary"):
                feature_data = {
                    "Feature": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                               "Insulin", "BMI", "Diabetes Pedigree", "Age"],
                    "Value": [pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age]
                }
                st.table(feature_data)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Add sidebar with information
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    This app predicts the likelihood of diabetes based on health metrics.
    
    **Input Features:**
    - Pregnancies
    - Glucose level
    - Blood pressure
    - Skin thickness
    - Insulin level
    - BMI
    - Diabetes pedigree function
    - Age
    
    **Note:** This is for educational purposes only. 
    Always consult healthcare professionals for medical advice.
    """)
    
    st.markdown("---")
    st.markdown("**Model Status:**")
    if model and hasattr(model, 'coef_'):
        st.success("✓ Real model loaded")
    else:
        st.warning("⚠️ Demo mode (using dummy model)")