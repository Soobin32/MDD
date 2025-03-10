from flask import Flask, request, jsonify # Flask: framework to create a web application, request: handles incoming HTTP requests, jsonify: converts python dictionires into JSON responses
# Flask allows to build web applications and APIs - helps python interact with web
# API (application programming interface) is a way of different programs (e.g. web) to communicate with each other - websites to send data to python, python to process and return results
# Flask API: receives HTTP requests (e.g. POST or GET) -> processes data -> returns a response
# HTTP request: message sent from browser (frontend) to the server (backend) - needed because website alone (HTML+JavaScript) can't run Python, so need a Python backend (Flask API) and this allows website to communicate with python
# JSON (JavaScript Object Notation): easy to read and used in APIs, JS naturally handles JSON, used for data transfer between JavaScript (frontend) and Python (Backend)
from flask_cors import CORS 
import pandas as pd # handles tabular data
import numpy as np # for numerical operations
import tensorflow as tf # loads and runs ML model
import os

# 🛑 Disable GPU usage to prevent CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "0"

# Create a Flask application instance (initialisation)
app = Flask(__name__)
# Set up CORS for allowing specific origins (for your HTML served on 127.0.0.1:8000)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["https://sleepykeepy.com", "https://mdd-k92n.onrender.com"]}})

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

# Processing Input Data and Making Predictions
def process_and_predict(data): # takes input sensor data, preprocesses it, and runt it through the model
    """Preprocess data and predict using the ML model."""
    print("Processing data: ", data)
    try:
        # Convert JSON data to DataFrame
        df = pd.DataFrame(data)

        # Ensure all data is numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Identify the two time reversal points
        reversal_indices = []
        for i in range(1, len(df)):
            if df["time"].iloc[i] < df["time"].iloc[i - 1]:  # Time went backwards
                reversal_indices.append(i)

        # Ensure we found exactly **two** reversal points
        if len(reversal_indices) != 2:
            return {"error": f"Expected exactly 2 time reversal points, found {len(reversal_indices)}"}

        # Split the data at the detected reversal points
        df_hr = df.iloc[:reversal_indices[0]].copy()  # Before first reversal → HR
        df_spo2 = df.iloc[reversal_indices[1]:].copy()  # After second reversal → SpO2
        df_breathing = df.iloc[reversal_indices[0]:reversal_indices[1]].copy()  # Between reversals → Breathing

        # Rename columns for merging
        df_hr.rename(columns={"value": "HR"}, inplace=True)
        df_spo2.rename(columns={"value": "SpO2"}, inplace=True)
        df_breathing.rename(columns={"value": "Breathing"}, inplace=True)

        # **MERGE WITHOUT CONSIDERING TIMESTAMPS** - Align row-wise
        min_len = min(len(df_hr), len(df_spo2), len(df_breathing))  # Trim to shortest length
        df_hr = df_hr.iloc[:min_len]  # Trim to match
        df_spo2 = df_spo2.iloc[:min_len]
        df_breathing = df_breathing.iloc[:min_len]

        # Combine values into a new DataFrame
        merged_df = pd.DataFrame({
            "HR": df_hr["HR"].values,
            "Breathing": df_breathing["Breathing"].values,  # Originally column 2
            "SpO2": df_spo2["SpO2"].values  # Originally column 3
        })

        # **Swap the SpO2 and Breathing columns**
        merged_df = merged_df[["HR", "SpO2", "Breathing"]]

        # Convert to NumPy array
        values = merged_df.to_numpy()

        # Reshape to (batch_size, 3, 1) for the model
        values = values.reshape(values.shape[0], 3, 1)

        # 🔍 Debug: Print first few merged and swapped values
        print("✅ Merged and Swapped input data (first 5 samples):")
        print(values[:5])

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

@app.route('/predict', methods=['POST']) # define an API endpoint /predict that listens for POST requests
# API endpoint: URL that a frontend can call to request data from a backend
# POST means this endpoint accepts data (GET - fetch data, PUT - update existing data, DELETE - remove data)
# in app.py, frontend sends sensor data 
def predict():
    if request.method == 'OPTIONS':
        return '', 200 # Respond OK to preflight request
    try:
        data = request.get_json() # extracts JSON data from the HTTP request
        print("Received data:", data)

        result = process_and_predict(data) # calls the function to generate predictions
        
        print("Prediction result", result)
        return jsonify(result) # returns the result as JSON

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)})

@app.route('/')
def home():
    return "API is running!", 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(debug=True, host="0.0.0.0", port=port)
    # runs the Flask App - starts the Flask server, waits for requests to come in, when a request is received, Flask routes it to the correct function
    # app.run run whatever was intiated with @app.route
    # post request made to /prediction -> runs predict() function
