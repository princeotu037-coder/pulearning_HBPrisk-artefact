from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load the trained Biased SVM model, scaler, and alpha
model = joblib.load("biased_svm_model.pkl")
scaler = joblib.load("scaler.pkl")
alpha_hat = joblib.load("alpha_hat.pkl")

@app.route('/')
def home():
    return "HBP Risk Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Get JSON data from request
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)  # Ensure correct shape

        # ✅ Check feature consistency
        if features.shape[1] != scaler.n_features_in_:
            return jsonify({"error": f"Expected {scaler.n_features_in_} features, but got {features.shape[1]}"}), 400

        # ✅ Scale the input
        features_scaled = scaler.transform(features)

        # ✅ Predict HBP risk
        probability = model.predict_proba(features_scaled)[:, 1] / alpha_hat
        probability = np.clip(probability, 0, 1)

        # ✅ Categorize risk level
        if probability >= 0.8:
            risk_level = "High Risk"
        elif probability >= 0.7:
            risk_level = "Medium Risk"
        elif probability >= 0.5:
            risk_level = "Low Risk"
        else:
            risk_level = "Very Low Risk"

        # ✅ Return response
        return jsonify({
            "HBP Risk Probability": round(float(probability[0]), 3),
            "Risk Level": risk_level
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
