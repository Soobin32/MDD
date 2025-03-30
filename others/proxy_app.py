import os
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, Bidirectional, GRU, Dense, Flatten,
    BatchNormalization, ReLU, Add, MaxPooling1D, Attention
)
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tabulate import tabulate
import time


# 🛑 Disable GPU usage to prevent CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "0"

# Create a single Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

print("Flask app is starting...")

# ----------------- Focal Loss -----------------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

try:
    # Log before loading the model
    print("Attempting to load model...")

    # Load the pre-trained model for apnea detection
    model = tf.keras.models.load_model("window20_fold_6.h5", custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25)}) 

    # Log after loading the model
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None in case of failure

import os

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TOKEN_URL = "https://api2.arduino.cc/iot/v1/clients/token"

HR_SPO2_THING_ID = os.getenv("HR_SPO2_THING_ID")
STRAIN_THING_ID = os.getenv("STRAIN_THING_ID")

HR_SPO2_PROPERTY_IDS = {
    "heartrate": os.getenv("HR_SPO2_HEARTRATE_ID"),
    "spO2": os.getenv("HR_SPO2_SPO2_ID")
}
STRAIN_PROPERTY_ID = os.getenv("STRAIN_PROPERTY_ID")


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
    """Fetch historical sensor data from Arduino Cloud API with pagination."""
    token = get_token()
    if not token:
        return jsonify({"error": "Failed to retrieve a valid token"}), 401

    now = datetime.utcnow()
    end_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Fetch data in smaller time chunks to bypass the 1000-record limit
    time_chunks = []
    chunk_size_hours = 12  # Fetch in 12-hour chunks (adjust if necessary)
    
    for i in reversed(range(14)):  # 14 chunks of 12 hours = 7 days
        chunk_start = now - timedelta(hours=(i + 1) * chunk_size_hours)
        chunk_end = now - timedelta(hours=i * chunk_size_hours)
        time_chunks.append((chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"), chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")))

    print("✅ Generated Time Chunks (Oldest → Newest):")
    for start, end in time_chunks:
        print(f"⏳ {start} → {end}")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    historical_data = {"heartrate": [], "spO2": [], "strain": []}

    # ✅ Function for paginated data fetching
    def fetch_paginated_data(variable, property_id, chunk_start, chunk_end):
        """Fetches paginated sensor data from Arduino Cloud API."""
        all_records = []
        next_from = chunk_start  # Start fetching from the beginning of the chunk

        while True:
            response = requests.get(
                f"https://api2.arduino.cc/iot/v2/things/{HR_SPO2_THING_ID}/properties/{property_id}/timeseries",
                headers=headers,
                params={"from": next_from, "to": chunk_end, "interval": 5, "limit": 1000}  # Fetch max 1000 per request
            )

            if response.status_code == 200:
                data = response.json()
                records = [{"time": entry["time"], "value": entry["value"]} for entry in data.get("data", [])]

                # ✅ Append newly fetched data
                all_records.extend(records)

                # Debugging: Print progress
                print(f"✅ {variable} - Retrieved {len(records)} points from {next_from} to {chunk_end}")

                # ✅ If fewer than 1000 records were returned, it means we got the last chunk. Stop fetching.
                if len(records) < 1000:
                    break  

                # ✅ Otherwise, update `next_from` to **one second after** the last retrieved timestamp
                last_time = records[-1]["time"]
                next_from = (datetime.fromisoformat(last_time.replace("Z", "")) + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

            else:
                print(f"⚠️ Failed to fetch {variable} for {chunk_start} - {chunk_end}: {response.text}")
                break  # Stop fetching on error

        return all_records

    # Fetch Heart Rate & SpO2 in paginated chunks
    for variable, property_id in HR_SPO2_PROPERTY_IDS.items():
        for chunk_start, chunk_end in time_chunks:
            historical_data[variable].extend(fetch_paginated_data(variable, property_id, chunk_start, chunk_end))

    # Fetch Breathing Strain in paginated chunks
    for chunk_start, chunk_end in time_chunks:
        historical_data["strain"].extend(fetch_paginated_data("strain", STRAIN_PROPERTY_ID, chunk_start, chunk_end))

    # ✅ Print final data counts
    print("✅ Final Data Sizes:")
    for key, value in historical_data.items():
        print(f"📊 {key}: {len(value)} records")

    return jsonify(historical_data)

def consecutive_occurrence(unfiltered_count, min_consecutive_gap = 20):
  # Now apply the consecutive occurrence filtering based on the gap
  count = unfiltered_count.copy()
  filtered_apnea_index = []
  previous_event = None
  for i in range(len(count)):
      if previous_event is None:
          filtered_apnea_index.append(count[i])
          previous_event = count[i]
      elif count[i] - previous_event >= min_consecutive_gap:
          filtered_apnea_index.append(count[i])
          previous_event = count[i]

  # Print the final filtered apnea events
  print(f"After filtering consecutive events with a gap < {min_consecutive_gap}s:")
  print(f"Filtered Apnea occurred {len(filtered_apnea_index)} times")
  print(f"Filtered Apnea occurred at: {str(filtered_apnea_index)} seconds from start time.")

  return filtered_apnea_index

def map_to_full_range(targets, indices, window_size=10, step=3, total_length=None):
    # If total_length is None, fall back to the default size
    if total_length is None:
        total_length = len(targets) * window_size

    # Initialize an array for the full range of time steps, initially set to zero
    full_targets = np.zeros(total_length)

    # For each window, assign the prediction to the correct positions in the full array
    for i, target in zip(indices, targets):
        full_targets[i:i+window_size] = target  # Map to corresponding window

    return full_targets


# ✅ Process Input Data and Make Predictions
def process_and_predict(data):
    """Preprocess sensor data & run it through the ML model."""
    print("Processing data: ", data)
    try:
        # Extract each sensor dataset separately
        df_hr = pd.DataFrame(data.get("heartrate", []))
        df_spO2 = pd.DataFrame(data.get("spO2", []))
        df_strain = pd.DataFrame(data.get("strain", []))

        # Ensure "value" column exists in all DataFrames
        for df in [df_hr, df_spO2, df_strain]:
            if "value" not in df:
                raise ValueError("Missing 'value' column in one of the datasets.")

        # Ensure all data is numeric
        df_hr["value"] = pd.to_numeric(df_hr["value"], errors="coerce")
        df_spO2["value"] = pd.to_numeric(df_spO2["value"], errors="coerce")
        df_strain["value"] = pd.to_numeric(df_strain["value"], errors="coerce")

        # Compute averages
        avg_hr = df_hr["value"].mean() if not df_hr.empty else None
        avg_spO2 = df_spO2["value"].mean() if not df_spO2.empty else None
        
        # Ensure equal lengths before merging
        min_len = min(len(df_hr), len(df_spO2), len(df_strain))
        df_hr, df_spO2, df_strain = df_hr.iloc[:min_len], df_spO2.iloc[:min_len], df_strain.iloc[:min_len]

        # Merge into a single DataFrame
        X_raw = pd.DataFrame({
            "HR": df_hr["value"].values,
            "SpO2": df_spO2["value"].values,
            "Breathing": df_strain["value"].values
        })

        window_size = 10
        step = 3

        X_windows = []
        indices = []

        for i in range(0, len(X_raw) - window_size + 1, step):
            X_window = X_raw[i:i+window_size]
            X_windows.append(X_window)
            indices.append(i)

        X = np.array(X_windows).astype(np.float32)

        # Debugging: Print first 5 rows
        print("✅ Before applying window (First 5 Samples):")
        print(X_raw.head())

        print("Model prediction started...")
        # Run model prediction
        prediction = model.predict(X)
        print("Model prediction completed.")

        # Convert probabilities to binary classification - 1=apnea, 0=normal
        pred_binary = np.squeeze((prediction > 0.45).astype(int))

        # Map predicted data back to the full range
        y_pred_full_range = map_to_full_range(pred_binary, indices, total_length = len(X_raw))

        # Post-process results
        pred_label = y_pred_full_range.copy()

        # Total duration of apnea
        apnea_moments = sum(pred_label)
        print("Total apnea moments: " + str(apnea_moments) + " s")

        # How many apnea events
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
        print("Count before consecutive occurrence filtering: "+ apnea_event_count)

        filtered_index = consecutive_occurrence(apnea_index)
        filtered_count = len(filtered_index)

        # Determine chest movement status
        chest_status = "Normal" if filtered_count <= 30 else "Abnormal"

        # Return results including average values
        return {
            "total_apnea_events": int(filtered_count),
            "average_heart_rate": round(avg_hr, 1) if avg_hr is not None else "No Data",
            "average_spO2": round(avg_spO2, 1) if avg_spO2 is not None else "No Data",
            "chest_movement": chest_status
        }

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
