from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load all models from models/ folder
models = {}
MODEL_FOLDER = "models"

if not os.path.exists(MODEL_FOLDER):
    raise FileNotFoundError(f"'{MODEL_FOLDER}' folder not found. Upload your model(s) here.")

for fname in os.listdir(MODEL_FOLDER):
    if fname.endswith(".pkl"):
        model_name = fname.replace(".pkl", "")
        models[model_name] = joblib.load(os.path.join(MODEL_FOLDER, fname))

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ SuccessLG API running on Heroku"})

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Request must include 'features' key"}), 400

        features = np.array(data["features"]).reshape(1, -1)
        prediction = models[model_name].predict(features)
        return jsonify({"model": model_name, "prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Only used for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
