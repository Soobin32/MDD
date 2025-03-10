# sets up a Flask API that fetches real-time sensor data from Arduino IoT Cloud
import base64
import json
import os
import time
from flask import Flask, request, jsonify # web framework
from flask_cors import CORS # enables cross-origin access - allows frontend to request backend
# CORS (Cross-origin Resource Sharing): security feature implemented by web browsers
# to prevent unauthorised websites from making requests to a different domain (a.k.a. cross-origin requests)
# A browser will block requests from one origin (frontend website) to another origin (backend server) unless the backend explicitly allows it
# frontend: user interface that people interacts with e.g. HTML, CSS, JavaScript; 
# backend: server-side logic that processes request e.g. Python, Java
# frontend sends HTTP requests to the backend, backend processes the request and returns data
import requests # makes HTTP requests to external APIs
import datetime
import json
import subprocess  # To run get_token.py
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app, supports_credentials=True, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


# Arduino API credentials
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

# Store token and expiration globally
cached_token = None
token_expiration = 0


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
        token_expiration = time.time() + expires_in - 60  # Refresh token 1 min early
        print("🔑 Token fetched successfully!")
        return cached_token
    else:
        print(f"❌ Failed to fetch token: {response.text}")
        return None


@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    """Fetch historical sensor data from Arduino Cloud API."""
    token = get_token()
    if not token:
        return jsonify({"error": "Failed to retrieve a valid token"}), 401

    print("🔑 Using Token:", token[:30], "... (truncated)")  # Debug print

    now = datetime.utcnow()
    start_time = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    historical_data = {}

    # Fetch data for heartrate and SpO2
    for variable, property_id in HR_SPO2_PROPERTY_IDS.items():
        url = f"https://api2.arduino.cc/iot/v2/things/{HR_SPO2_THING_ID}/properties/{property_id}/timeseries"

        response = requests.get(url, headers=headers, params={"from": start_time, "to": end_time, "interval": 5})
        if response.status_code == 200:
            data = response.json()
            historical_data[variable] = [{"time": entry["time"], "value": entry["value"]} for entry in data.get("data", [])]
        else:
            historical_data[variable] = {"error": f"Failed to fetch data: {response.text}", "status_code": response.status_code}

    # Fetch data for breathing strain
    strain_url = f"https://api2.arduino.cc/iot/v2/things/{STRAIN_THING_ID}/properties/{STRAIN_PROPERTY_ID}/timeseries"
    strain_response = requests.get(strain_url, headers=headers, params={"from": start_time, "to": end_time, "interval": 5})

    if strain_response.status_code == 200:
        strain_data = strain_response.json()
        historical_data["strain"] = [{"time": entry["time"], "value": entry["value"]} for entry in strain_data.get("data", [])]
    else:
        historical_data["strain"] = {"error": f"Failed to fetch strain data: {strain_response.text}", "status_code": strain_response.status_code}

    return jsonify(historical_data)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port or fallback to 5000 locally
    app.run(debug=True, host="0.0.0.0", port=port)
