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

pulearning_HBPrisk-artefact/
│── hbp-flask-backend/  # Backend (Flask API)
│   ├── app.py          # Flask API
│   ├── biased_svm_model.pkl  # Trained Model
│   ├── scaler.pkl      # StandardScaler object
│   ├── alpha_hat.pkl   # Class Prior Estimation
│   ├── requirements.txt # Dependencies
│   ├── Procfile        # Render Deployment Config
│── hbp-frontend_1/     # Frontend (React)
│   ├── src/            # React Source Code
│   │   ├── components/ # UI Components
│   │   ├── styles/     # CSS Files
│   │   ├── App.jsx     # Main App
│   │   ├── main.jsx    # React DOM Entry
│   ├── index.html      # Main HTML
│   ├── package.json    # Frontend Dependencies
│── README.md           # Project Documentation


---

## 🛠 Installation & Setup

### **1️⃣ Install Dependencies**
Make sure you have **Python 3.8+** installed, then install the required packages:

```sh
pip install -r requirements.txt


---
