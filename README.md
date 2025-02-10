# **HBP Risk Prediction – PU Learning Model**  

[![API Status](https://img.shields.io/website?url=https://pulearning-hbp-risk.onrender.com)](https://pulearning-hbp-risk.onrender.com) 
[![Frontend Status](https://img.shields.io/website?url=https://pulearning-hbprisk-artefact.pages.dev)](https://pulearning-hbprisk-artefact.pages.dev)  
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)  

## **Overview**  

This project presents a **Hypertension (HBP) Risk Prediction System** using **Positive-Unlabeled (PU) Learning**. The system consists of:  

- **Machine Learning Model** – Biased SVM trained to predict hypertension risk.  
- **🖥Backend (Flask API)** – Serves real-time predictions.  
- **Frontend (React + Tailwind CSS)** – Web-based interface for user-friendly input.  
- **Deployment** – Flask API on **Render**, Frontend on **Cloudflare Pages**.  

🚀 **Live Demo:**  
- **Frontend**: [HBP Risk Prediction](https://pulearning-hbprisk-artefact.pages.dev/)  
- **API Endpoint**: [HBP Risk API](https://pulearning-hbp-risk.onrender.com)  

---

## **Tech Stack**  

| Component  | Technology Used  |
|------------|----------------|
| **Machine Learning** | Python, Scikit-learn (Biased SVM) |
| **Backend** | Flask, Flask-CORS, Gunicorn |
| **Frontend** | React.js, Tailwind CSS |
| **Deployment** | Render (API), Cloudflare Pages (Frontend) |
| **Version Control** | Git & GitHub |



---

## **🔬 Machine Learning Approach**  

### **1️⃣ PU Learning (Biased SVM Model)**  
- Instead of traditional supervised learning, this model learns from **Positive (HBP cases)** and **Unlabeled (PU) data**, where negative cases are uncertain.  
- **Biased SVM** assigns lower importance to unlabeled data to improve classification accuracy.  

### **2️⃣ Feature Engineering**  
- Inputs include **12 medical parameters** such as **Age, BMI, BP, Cholesterol, Glucose, and Heart Rate**.  
- Data is **standardized** using **StandardScaler** to ensure optimal model performance.  

### **3️⃣ Risk Prediction & Categorization**  
- The model outputs a **Hypertension Risk Probability**.  
- The **Risk Levels** are categorized as:  
  - **Very Low Risk** (< 0.5)  
  - **Mild Risk** (0.5 – 0.7)  
  - **Moderate Risk** (0.7 – 0.8)  
  - **High Risk** (> 0.8)  

---

## **🚀 Deployment & API Usage**  

### **1️⃣ Flask API Deployment (Render)**  
- The backend is deployed on **Render** with the API endpoint:  
  **`https://pulearning-hbp-risk.onrender.com/predict`**  
- The API accepts **POST requests** with JSON data in this format:  
  ```json
  {
      "features": [1, 39, 27, 0, 0, 0, 195, 115, 106, 80, 70, 77]
  }
 

