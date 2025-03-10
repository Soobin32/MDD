import os
import base64
import json
import time
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
from datetime import datetime, timedelta

# 🛑 Disable GPU usage to prevent CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "0"

# Create a single Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

print("Flask app is starting...")

try:
    # Log before loading the model
    print("Attempting to load model...")

    # Load the pre-trained model for apnea detection
    model = tf.keras.models.load_model("model3_fold_2.h5") 

    # Log after loading the model
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None in case of failure

# ✅ Arduino API Credentials
CLIENT_ID = "s6SgKndbcK7pC7FyawZRJT1fRV4GPuFD"
CLIENT_SECRET = "Gyt3GzszTd5MfqHtPEsA9swXnyqU837C41NKYP710UzcA3dKcP64PtruURShD0zZ"
TOKEN_URL = "https://api2.arduino.cc/iot/v1/clients/token"

# Arduino API endpoint details
HR_SPO2_THING_ID = "8fa4357b-69cb-4d75-bbd8-974dddb880f9"
STRAIN_THING_ID = "e9b06dd0-70c3-41bf-bfd7-a9d84b985255"

HR_SPO2_PROPERTY_IDS = {
    "heartrate": "d0e35245-cbbb-4676-9021-9b0612e4c674",
    "sp02": "81d9530f-ec85-4dce-97a9-c04f1cfad0b5"
}
STRAIN_PROPERTY_ID = "80951354-1c52-476c-bb1f-d8e83624ae96"

cached_token = None
token_expiration = 0  # Store token and expiration globally


# ✅ Fetch Arduino API Token
def get_token():
    """Fetches a new token from Arduino API if expired."""
    global cached_token, token_expiration

    # If token is still valid, return it
    if cached_token and time.time() < token_expiration:
        print("✅ Using cached token")
        return cached_token

    # Fetch new token
    print("🔄 Fetching a new token from Arduino API...")
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "audience": "https://api2.arduino.cc/iot"
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    if response.status_code == 200:
        response_json = response.json()
        cached_token = response_json.get("access_token", "")
        expires_in = response_json.get("expires_in", 3600)
        token_expiration = time.time() + expires_in - 60
        print("🔑 Token fetched successfully!")
        return cached_token
    else:
        print(f"❌ Failed to fetch token: {response.text}")
        return None


# ✅ Fetch Sensor Data from Arduino IoT
@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    """Fetch historical sensor data from Arduino Cloud API."""
    token = get_token()
    if not token:
        return jsonify({"error": "Failed to retrieve a valid token"}), 401

    now = datetime.utcnow()
    start_time = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    historical_data = {}

    # Fetch Heart Rate & SpO2
    for variable, property_id in HR_SPO2_PROPERTY_IDS.items():
        url = f"https://api2.arduino.cc/iot/v2/things/{HR_SPO2_THING_ID}/properties/{property_id}/timeseries"
        response = requests.get(url, headers=headers, params={"from": start_time, "to": end_time, "interval": 5})
        if response.status_code == 200:
            data = response.json()
            historical_data[variable] = [{"time": entry["time"], "value": entry["value"]} for entry in data.get("data", [])]
        else:
            historical_data[variable] = {"error": f"Failed to fetch data: {response.text}", "status_code": response.status_code}

    # Fetch Breathing Strain
    strain_url = f"https://api2.arduino.cc/iot/v2/things/{STRAIN_THING_ID}/properties/{STRAIN_PROPERTY_ID}/timeseries"
    strain_response = requests.get(strain_url, headers=headers, params={"from": start_time, "to": end_time, "interval": 5})

    if strain_response.status_code == 200:
        strain_data = strain_response.json()
        historical_data["strain"] = [{"time": entry["time"], "value": entry["value"]} for entry in strain_data.get("data", [])]
    else:
        historical_data["strain"] = {"error": f"Failed to fetch strain data: {strain_response.text}", "status_code": strain_response.status_code}

    return jsonify(historical_data)


# ✅ Process Input Data and Make Predictions
def process_and_predict(data):
    """Preprocess sensor data & run it through the ML model."""
    print("Processing data: ", data)
    try:
        # Convert JSON data to DataFrame
        df = pd.DataFrame(data)

        # Ensure all data is numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Identify the two time reversal points
        reversal_indices = [i for i in range(1, len(df)) if df["time"].iloc[i] < df["time"].iloc[i - 1]]
        # Ensure we found exactly **two** reversal points
        if len(reversal_indices) != 2:
            return {"error": f"Expected exactly 2 time reversal points, found {len(reversal_indices)}"}

        # Split the data at the detected reversal points
        df_hr = df.iloc[:reversal_indices[0]].copy()
        df_spo2 = df.iloc[reversal_indices[1]:].copy()
        df_breathing = df.iloc[reversal_indices[0]:reversal_indices[1]].copy()

        # Rename columns for merging
        df_hr.rename(columns={"value": "HR"}, inplace=True)
        df_spo2.rename(columns={"value": "SpO2"}, inplace=True)
        df_breathing.rename(columns={"value": "Breathing"}, inplace=True)

        # **MERGE WITHOUT CONSIDERING TIMESTAMPS** - Align row-wise
        min_len = min(len(df_hr), len(df_spo2), len(df_breathing))
        df_hr, df_spo2, df_breathing = df_hr.iloc[:min_len], df_spo2.iloc[:min_len], df_breathing.iloc[:min_len]

        # Combine values into a new DataFrame, Swap the SpO2 and Breathing columns**
        merged_df = pd.DataFrame({"HR": df_hr["HR"].values, "SpO2": df_spo2["SpO2"].values, "Breathing": df_breathing["Breathing"].values})
        values = merged_df.to_numpy().reshape(values.shape[0], 3, 1)

        print("Model prediction started...")
        # Make prediction i.e. run the model
        prediction = model.predict(values, batch_size=1)
        print("Model prediction completed.")

        # Convert probability to binary classification - 1=apnea, 0=normal
        pred_binary = np.squeeze((prediction > 0.5).astype(int))

        # Post-process results
        pred_label = pred_binary.copy()
        previous_value = 0

        for i in range(len(pred_label)):
            if pred_label[i] == 1:
                if previous_value == 1:
                    pred_label[i] = 0  # Change consecutive 1s to 0 - ensure onlny the first detection of each apnea event is counted
                previous_value = 1
            else:
                previous_value = 0

        # Count apnea events
        apnea_index = [i for i in range(len(pred_label)) if pred_label[i] == 1] # stores index positions of apnea events and counts them
        apnea_event_count = len(apnea_index)

        # Return results
        return {"total_apnea_events": apnea_event_count}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}


# ✅ Predict Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 200  # Preflight success
        
    try:
        data = request.get_json()
        print("Received data:", data)
        if not data:
            return jsonify({"error": "No data received"}), 400

        result = process_and_predict(data)
        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))  # Use Render's assigned port
