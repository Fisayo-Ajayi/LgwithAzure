from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# --- Load Models ---
# Ensure these files exist in the "models" directory at the project root
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
KMEANS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "kmeans.pkl")

scaler = joblib.load(SCALER_PATH)
kmeans = joblib.load(KMEANS_PATH)


# --- Routes ---
@app.route("/")
def home():
    return jsonify({
        "message": "✅ LG Project API is running on Vercel!",
        "endpoints": {
            "/predict": "POST features to get a cluster prediction"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request body"}), 400

        # Convert to numpy and scale
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Predict with KMeans
        cluster = kmeans.predict(features_scaled)[0]

        return jsonify({
            "cluster": int(cluster),
            "input_features": data["features"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


