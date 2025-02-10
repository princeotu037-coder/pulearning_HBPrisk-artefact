# **HBP Risk Prediction – PU Learning Model**  

[![API Status](https://img.shields.io/website?url=https://pulearning-hbp-risk.onrender.com)](https://pulearning-hbp-risk.onrender.com) 
[![Frontend Status](https://img.shields.io/website?url=https://pulearning-hbprisk-artefact.pages.dev)](https://pulearning-hbprisk-artefact.pages.dev) [![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)  

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

| **Component**  | Technology Used  |
|------------|----------------|
| **Machine Learning** | Python, Scikit-learn (Biased SVM) |
| **Backend** | Flask, Flask-CORS, Gunicorn |
| **Frontend** | React.js, Tailwind CSS |
| **Deployment** | Render (API), Cloudflare Pages (Frontend) |
| **Version Control** | Git & GitHub |



---

## **PU Learning (Biased SVM Model)**  

- Instead of traditional supervised learning, this model learns from **Positive (HBP cases)** and **Unlabeled (PU) data**, where negative cases are uncertain.  Weighted logistic pu learning as well as Pu bagging were trained but Biased svm showed robust results with prediction and was therefore chosen.
- **Biased SVM** assigns lower importance to unlabeled data to improve classification accuracy.  


---

## **🚀 Deployment & API Usage** 

### ** API Request
- The backend is deployed on **Render** with the API endpoint:  
  **`https://pulearning-hbp-risk.onrender.com/predict`**  
- The API accepts **POST requests** with JSON data in this format:  
  ```json
  {
      "features": [1, 52, 29, 0, 0, 0, 240, 115, 132, 84, 70, 77]
  }
  
### ** API Response **
```json
{
    "HBP Risk Probability": 0.81,
    "Risk Level": "High Risk (Hypertensive)"
}
