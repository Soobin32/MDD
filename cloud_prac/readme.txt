get_token.py (CGI Script) - Fetches Arduino API token and saves it for reuse.
proxy.py (Flask API) - Uses the token to fetch sensor data from Arduino Cloud.
app.py (Flask ML Model) - Processes sensor data and makes predictions using a TensorFlow model.
cloudsample.html (Frontend Dashboard) - Fetches sensor data from proxy.py, and sends it to app.py for predictions.