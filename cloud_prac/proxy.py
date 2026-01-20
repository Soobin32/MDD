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


app = Flask(__name__) # creates Flask app
CORS(app, supports_credentials=True, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
   # Allows browser access - frontend (browser) can access this backend

# Arduino API endpoint for fetching historical sensor data
HR_SPO2_THING_ID = "8fa4357b-69cb-4d75-bbd8-974dddb880f9"
STRAIN_THING_ID = "e9b06dd0-70c3-41bf-bfd7-a9d84b985255"  # Strain device's different THING_ID

# Property IDs for both variables
HR_SPO2_PROPERTY_IDS = {
    "heartrate": "d0e35245-cbbb-4676-9021-9b0612e4c674",  # Replace with heartrate Property ID
    "sp02": "81d9530f-ec85-4dce-97a9-c04f1cfad0b5"  # Replace with sp02 Property ID
}

STRAIN_PROPERTY_ID = "80951354-1c52-476c-bb1f-d8e83624ae96"  # Strain Property ID

TOKEN_FILE = "var/www/cgi-bin/token.txt"
GET_TOKEN_SCRIPT = "python3 var/www/cgi-bin/get_token.py"  # Command to run get_token.py

def generate_new_token():
    """Runs get_token.py to generate a new API token."""
    print("Running get_token.py to generate a new token...")
    subprocess.run(GET_TOKEN_SCRIPT, shell=True, check=True)
    time.sleep(2)  # Wait a bit for token to be written
    return load_token()  # Reload the token after generating

def load_token():
    """Load the API token from token.txt, check expiration, and refresh if needed."""
    if not os.path.exists(TOKEN_FILE):
        print("Token file does not exist! Running get_token.py...")
        return generate_new_token()

    try:
        with open(TOKEN_FILE, "r") as f:
            token = f.read().strip()
            if not token:
                print("Token file is empty! Running get_token.py...")
                return generate_new_token()

            # Decode JWT token to check expiration
            token_parts = token.split(".")
            if len(token_parts) != 3:
                print("Invalid token format! Running get_token.py...")
                return generate_new_token()

            payload = json.loads(base64.urlsafe_b64decode(token_parts[1] + "==").decode("utf-8"))
            exp_time = payload.get("exp", 0)

            if time.time() > exp_time:
                print("Token expired! Generating a new one...")
                return generate_new_token()

            print(f"Loaded Valid Token: {token[:30]}... (truncated)")
            return token

    except Exception as e:
        print(f"Error reading token: {str(e)}. Running get_token.py...")
        return generate_new_token()

    
# fetch data from arduino cloud
@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    """Fetch historical sensor data for all variables from Arduino Cloud API."""
    token = load_token()
    if not token:
        return jsonify({"error": "Failed to retrieve a valid token"}), 401

    print(f"Using Token: {token[:30]}... (truncated)")  # Debug print

    # Define the time range (last 30 days)
    now = datetime.utcnow()
    start_time = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")  # Last 30 days
    end_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")  # Current time

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    historical_data = {}

    # Fetch historical data for each property
    for variable, property_id in HR_SPO2_PROPERTY_IDS.items():
        url = f"https://api2.arduino.cc/iot/v2/things/{HR_SPO2_THING_ID}/properties/{property_id}/timeseries"

        params = {
            "from": start_time,
            "to": end_time,
            "interval": 5  # 5 sec interval
        }

        print(f"Sending API request to: {url}")
        print(f"Request Params: {params}")
        
        # Make the API request
        response = requests.get(url, headers=headers, params=params)

        # Debugging print: Print the API response body before processing
        print(f"API Response Body: {response.text}")  # Debugging print

        if response.status_code == 200:
            data = response.json()

            # Debugging print: Check if "data" and "values" exist in the response
            print(f"Data for {variable}: {data.get('data', [])}")  # Debugging print

            values = [
                {"time": entry["time"], "value": entry["value"]}
                for entry in data.get("data", [])
            ]
            historical_data[variable] = values
        else:
            historical_data[variable] = {
                "error": f"Failed to fetch data: {response.text}",
                "status_code": response.status_code
            }
    
    # Fetch historical data for strain (from STRAIN_THING_ID)
    strain_url = f"https://api2.arduino.cc/iot/v2/things/{STRAIN_THING_ID}/properties/{STRAIN_PROPERTY_ID}/timeseries"

    strain_params = {
        "from": start_time,
        "to": end_time,
        "interval": 5
    }

    print(f"Sending API request to: {strain_url}")
    print(f"Request Params: {strain_params}")

    # Make the API request from strain data
    strain_response = requests.get(strain_url, headers=headers, params=strain_params)

    if strain_response.status_code == 200:
        strain_data = strain_response.json()

        # Debugging print: Check if "data" and "values" exist in the response
        print(f"Data for strain: {strain_data.get('data', [])}")  # Debugging print

        strain_values = [
            {"time": entry["time"], "value": entry["value"]}
            for entry in strain_data.get("data", [])
        ]
        historical_data["strain"] = strain_values
    else:
        historical_data["strain"] = {
            "error": f"Failed to fetch strain data: {strain_response.text}",
            "status_code": strain_response.status_code
        }

    print(f"Historical Data: {historical_data}")  # Debugging print

    return jsonify(historical_data)


if __name__ == '__main__':
    app.run(debug=True, port=5001) # starts Flask app on port 5001
    # port = like a 'door' on a server that allows communication between different applications
    # default is 5000 but 5001 is used when you separate backend services (e.g. app.py on 5000, proxy.py on 5001)


# 'python proxy.py' before running
