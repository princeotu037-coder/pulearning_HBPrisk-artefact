# Load the saved Biased SVM model, scaler, and alpha value
loaded_model = joblib.load("biased_svm_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
loaded_alpha = joblib.load("alpha_hat.pkl")

print("Model, scaler, and alpha loaded successfully!")

# Simulating a new patient's data (replace with real input values)
# Ensure the number of features matches the training data
new_patient = np.array([[45, 130, 85, 1, 0, 1, 25.5, 1, 0, 1, 0, 1]])  # Adjusted to 12 features

# Scale the new patient's data
new_patient_scaled = loaded_scaler.transform(new_patient.reshape(1, -1))  # Reshape for compatibility

# Predict HBP risk probability
probability = loaded_model.predict_proba(new_patient_scaled)[:, 1] / loaded_alpha
probability = np.clip(probability, 0, 1)

# Categorize risk level
if probability >= 0.8:
    risk_level = "High Risk"
elif probability >= 0.7:
    risk_level = "Medium Risk"
elif probability >= 0.5:
    risk_level = "Low Risk"
else:
    risk_level = "Very Low Risk"

print(f"Predicted HBP Risk: {probability[0]:.3f} ({risk_level})")
