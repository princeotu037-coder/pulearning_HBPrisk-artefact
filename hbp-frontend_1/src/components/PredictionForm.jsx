import React, { useState } from "react";

const PredictionForm = () => {
  const [features, setFeatures] = useState({
    Sex: 1,
    Age: 39,
    BMI: 26.97,
    ExxAngina: 0,
    BPMeds: 0,
    Diabetes: 0,
    Cholesterol: 195,
    RestBP: 115,
    SystBP: 106,
    DiastBP: 70,
    HeartRate: 80,
    Glucose: 77,
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFeatures({ ...features, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        "https://thingproxy.freeboard.io/fetch/https://pulearning-hbp-risk.onrender.com/predict",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ features: Object.values(features) }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to get a response from API");
      }

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto p-6 bg-white shadow-lg rounded-lg">
      <h2 className="text-xl font-bold text-gray-800">HBP Risk Prediction</h2>
      <form onSubmit={handleSubmit} className="mt-4 space-y-3">
        {Object.keys(features).map((key) => (
          <div key={key}>
            <label className="block text-sm font-medium text-gray-700">
              {key}
            </label>
            <input
              type="number"
              name={key}
              value={features[key]}
              onChange={handleChange}
              className="mt-1 p-2 w-full border rounded-md"
              required
            />
          </div>
        ))}

        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600"
          disabled={loading}
        >
          {loading ? "Predicting..." : "Get Risk Level"}
        </button>
      </form>

      {error && <p className="text-red-500 mt-4">{error}</p>}
      {prediction && (
        <p className="mt-4 text-lg font-semibold text-green-600">
          Prediction: {prediction}
        </p>
      )}
    </div>
  );
};

export default PredictionForm;
