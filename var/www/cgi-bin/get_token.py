#!/usr/bin/env python3

import cgi
import json
import requests

# Arduino API credentials
CLIENT_ID = "s6SgKndbcK7pC7FyawZRJT1fRV4GPuFD"
CLIENT_SECRET = "Gyt3GzszTd5MfqHtPEsA9swXnyqU837C41NKYP710UzcA3dKcP64PtruURShD0zZ"
TOKEN_URL = "https://api2.arduino.cc/iot/v1/clients/token"

# Prepare API request
data = {
    "grant_type": "client_credentials",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "audience": "https://api2.arduino.cc/iot"
}

# Make request to Arduino API
response = requests.post(TOKEN_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})

# Print headers for CGI
print("Content-Type: application/json")
print("Access-Control-Allow-Origin: *")
print("")

# Process response
if response.status_code == 200:
    response_json = response.json()
    access_token = response_json.get("access_token", "")

    # Save token to a file
    with open("var/www/cgi-bin/token.txt", "w") as f:
        f.write(access_token)

    # Return token as JSON
    print(json.dumps({"access_token": access_token}))
else:
    print(json.dumps({"error": "Failed to fetch token", "status_code": response.status_code}))
