import React, { useState } from "react";
import axios from "axios";

const AnalysisForm = ({ onAnalysisComplete, onError }) => {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    console.log("Submitting text for analysis:", text);
    try {
      const response = await axios.post("http://localhost:8000/analyze", {
        text,
      });
      console.log("Analysis response:", response.data);
      onAnalysisComplete(response.data);
    } catch (err) {
      console.error("Error analyzing text:", err);
      const errorMessage =
        err.response?.data?.detail || "Failed to analyze text";
      setError(errorMessage);
      onError(new Error(errorMessage));
    } finally {
      setLoading(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{ width: "100%", marginBottom: "20px" }}
    >
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text for analysis"
        rows={5}
        style={{
          width: "100%",
          padding: "10px",
          borderRadius: "5px",
          border: "1px solid #ccc",
        }}
        disabled={loading}
      />
      {error && <div style={{ color: "red", marginTop: "10px" }}>{error}</div>}
      <button
        type="submit"
        style={{
          marginTop: "10px",
          padding: "10px 20px",
          backgroundColor: "#4CAF50",
          color: "#fff",
          border: "none",
          borderRadius: "5px",
          cursor: loading ? "not-allowed" : "pointer",
          opacity: loading ? 0.6 : 1,
        }}
        disabled={loading}
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>
    </form>
  );
};

export default AnalysisForm;
