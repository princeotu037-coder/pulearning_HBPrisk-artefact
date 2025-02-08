from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("biased_svm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/", methods=["GET"])
def home():
    return "HBP Risk Prediction API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    print(f"Incoming request method: {request.method}")  # 🔍 Debugging
    print(f"Incoming request JSON: {request.json}")  # 🔍 Debugging
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[:, 1][0]

        if probability < 0.7:
            risk_category = "Low Risk"
        elif probability < 0.8:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"

        return jsonify({"risk_category": risk_category, "probability": round(probability, 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
