from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('model/mace_ml_model.pkl')
scaler = joblib.load('model/scaler.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')  # Load the frontend


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        features = [float(x) for x in request.form.values()]
        features_scaled = scaler.transform([features])  # Normalize input
        prediction = model.predict(features_scaled)[0]  # Predict

        result = "Gamma-Ray Event" if prediction == 1 else "Hadron Event"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
