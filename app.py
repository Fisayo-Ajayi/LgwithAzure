from fastapi import FastAPI, Request
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "✅ LG Customer Segmentation API is live on Hugging Face!"}

@app.post("/segment")
async def segment(request: Request):
    try:
        data = await request.json()

        required_features = [
            "Age","Income","LoyaltyScore","OnlineEngagement",
            "DaysSinceLastPurchase","QuantityPurchased",
            "PreferenceScore","WillingnessToPay"
        ]

        # Check required features
        if not all(f in data for f in required_features):
            missing = [f for f in required_features if f not in data]
            return {"error": f"Missing features: {missing}"}

        # Convert input to DataFrame
        features = pd.DataFrame([{
            "Age": data["Age"],
            "Income": data["Income"],
            "LoyaltyScore": data["LoyaltyScore"],
            "OnlineEngagement": data["OnlineEngagement"],
            "DaysSinceLastPurchase": data["DaysSinceLastPurchase"],
            "QuantityPurchased": data["QuantityPurchased"],
            "PreferenceScore": data["PreferenceScore"],
            "WillingnessToPay": data["WillingnessToPay"]
        }])

        # Scale + predict
        features_scaled = scaler.transform(features)
        cluster = kmeans.predict(features_scaled)[0]

        return {"CustomerSegment": int(cluster)}

    except Exception as e:
        return {"error": str(e)}
