from flask import Flask, render_template, request, jsonify
import joblib
import os
import sys

# Add the parent directory to Python path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Create the Flask app
app = Flask(__name__)

# Define the expected input fields
fields = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
          'insulin', 'bmi', 'dpf', 'age']

# Load the model and scaler using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/diabetes_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '../models/scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Scaler path: {SCALER_PATH}")
    # Create dummy model and scaler for testing if files don't exist
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    model = LogisticRegression()
    scaler = StandardScaler()
    # Fit with dummy data
    dummy_X = np.random.rand(10, 8)
    dummy_y = np.random.randint(0, 2, 10)
    scaler.fit(dummy_X)
    model.fit(dummy_X, dummy_y)
    print("Using dummy model for testing")

# Health check endpoint for Docker/cloud deployment
@app.route('/health')
def health():
    try:
        # Check if model and scaler are loaded
        model_status = "loaded" if hasattr(model, 'predict') else "dummy"
        scaler_status = "loaded" if hasattr(scaler, 'transform') else "dummy"
        
        return jsonify({
            'status': 'healthy',
            'message': 'Diabetes Prediction API is running',
            'model_status': model_status,
            'scaler_status': scaler_status,
            'timestamp': os.times().user  # Simple timestamp
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Home route - shows the input form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route - triggered by form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []

        # Collect and validate input fields
        for field in fields:
            value = request.form.get(field)
            if value is None or value.strip() == '':
                return render_template('index.html', error=f'Missing input: {field}')
            try:
                features.append(float(value))
            except ValueError:
                return render_template('index.html', error=f'Invalid input for: {field}')

        # Scale the input
        features_scaled = scaler.transform([features])

        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]

        # Prepare result dictionary
        result = {
            'prediction': int(prediction[0]),
            'probability': round(float(probability), 4),
            'message': 'Diabetes detected' if prediction[0] == 1 else 'No diabetes detected'
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('index.html', error=f'Prediction error: {str(e)}')

# Optional: API version of the prediction endpoint (for programmatic access)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        features = []

        for field in fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'})
            try:
                features.append(float(data[field]))
            except ValueError:
                return jsonify({'error': f'Invalid input for: {field}'})

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': round(float(probability), 4),
            'message': 'Diabetes detected' if prediction[0] == 1 else 'No diabetes detected'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Entry point
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)