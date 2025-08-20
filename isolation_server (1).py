from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS  # Allow Unity to connect

# Load dataset
dataset = pd.read_csv("Yonghe_Modal_FDD_with_headers.csv")  # Ensure the file is in the same folder

# Ensure we are selecting only numerical data (Mode 3)
mode3_all = dataset.iloc[2].values[1:].astype(float)

# Normalize the data using StandardScaler
scaler = StandardScaler()
mode3_scaled = scaler.fit_transform(mode3_all.reshape(-1, 1))

# Train Isolation Forest
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
isolation_forest.fit(mode3_scaled)

# Set up Flask API
app = Flask(__name__)
CORS(app)  # Allow Unity to connect

# Home route to check if server is running
@app.route('/')
def home():
    return "Flask Anomaly Detection API is running!"

# Handle missing favicon.ico requests (prevents 404 errors)
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content response



# Anomaly detection endpoint
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'GET':
            return "This endpoint requires a POST request with a frequency value.", 400
        # Get frequency data from Unity
        frequency = request.form.get("frequency")
        wind_strength = request.form.get("wind_strength") 

        if frequency is None:
            return jsonify({"error": "No frequency value provided"}), 400

        # Convert to float
        frequency = float(frequency)
        wind_strength = float(wind_strength)
        print(f"Received Frequency: {frequency}, Wind Strength: {wind_strength}")

        # Scale input frequency
        scaled_frequency = scaler.transform([[frequency]])
        
        # Predict anomaly (-1 = Anomaly, 1 = Normal)
        prediction = isolation_forest.predict(scaled_frequency)
        result = int(prediction[0])

        # Log and return the result
        print(f"Prediction: {'Anomaly' if result == -1 else 'Normal'}")
        return jsonify({
            "prediction": result,
            "frequency": frequency,
            "wind_strength": wind_strength
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run Flask Server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)  # Change port if needed
