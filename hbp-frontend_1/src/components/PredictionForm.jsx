import { useState } from "react";
import "./PredictionForm.css"; // 

const PredictionForm = () => {
  const [features, setFeatures] = useState([
    1, 52, 27, 0, 0, 0, 220, 115, 125, 70, 80, 77
  ]); // 

  const [result, setResult] = useState(null);

  const handleChange = (index, value) => {
    const updatedFeatures = [...features];
    updatedFeatures[index] = value;
    setFeatures(updatedFeatures);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = { features: features.map(Number) }; 

    try {
      const response = await fetch("https://pulearning-hbp-risk.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Failed to get prediction" });
    }
  };

  return (
    <div className="form-container">
      <h2>HBP Risk Prediction</h2>
      <p className="description">
        Enter patient details to predict hypertension risk. (Note: For ExxAngina, BPMeds, and Diabetes, 1 = Yes and 0 = No)
      </p>

      <form onSubmit={handleSubmit}>
        <div className="grid-container">
          {["Sex: 1 = Male, 0 = Female", "Age", "BMI: Weight over Sq. Height", "Exx Induced Angina?", "Takes BPMeds?", "Diabetic?", "Cholesterol Level", "Resting BP (Sys)", "Systolic BP", "Diastolic BP", "Heart Rate", "Glucose Level"].map(
            (label, index) => (
              <div key={index} className="input-box">
                <label>{label}</label>
                <input
                  type="number"
                  value={features[index]}
                  onChange={(e) => handleChange(index, e.target.value)}
                  required
                />
              </div>
            )
          )}
        </div>
        <button type="submit">Get Risk Level</button>
      </form>

      {result && (
        <div className="result-box">
          <h3>Prediction Result:</h3>
          <p>HBP Risk Probability: <strong>{result["HBP Risk Probability"]}</strong></p>
          <p>Risk Level: <strong>{result["Risk Level"]}</strong></p>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
