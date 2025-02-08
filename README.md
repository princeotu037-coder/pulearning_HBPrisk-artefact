# HBP Risk Prediction API

This project is a **Flask API** that predicts the risk of **Hypertension (HBP)** using a **Positive-Unlabeled (PU) Learning** model, specifically a **Biased SVM model**. Users can input patient details, and the API will classify the **HBP risk level** as:
- **Low Risk (0.5 - 0.7)**
- **Medium Risk (0.7 - 0.8)**
- **High Risk (0.8 - 1.0)**

## 🚀 Features
- **Machine Learning Model** trained using **Positive-Unlabeled (PU) Learning**.
- **RESTful API** built with Flask.
- **Model Persistence** with `joblib` for efficient loading.
- **Standardization** using `StandardScaler` for consistent predictions.
- **Deployment-Ready** for hosting on **Cloudflare Pages**.

---

## 📂 Project Structure

HBP-Risk-Prediction/ │── model/ # Contains trained Biased SVM model │ ├── scaler.pkl # StandardScaler for feature normalization │ ├── biased_svm.pkl # Saved Biased SVM model │ │── app.py # Flask API backend │── requirements.txt # Dependencies for the project │── README.md │── test_request.json # Sample input for Postman testing

---

## 🛠 Installation & Setup

### **1️⃣ Install Dependencies**
Make sure you have **Python 3.8+** installed, then install the required packages:

```sh
pip install -r requirements.txt


---
