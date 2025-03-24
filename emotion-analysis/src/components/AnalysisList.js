import React, { useEffect, useState } from "react";
import axios from "axios";

const AnalysisList = ({ onSelectAnalysis, onError }) => {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState("");

  useEffect(() => {
    const fetchAnalyses = async () => {
      setLoading(true);
      console.log("Fetching analyses...");
      try {
        const response = await axios.get("http://localhost:8000/analyses");
        console.log("Analyses fetched:", response.data);
        setAnalyses(response.data);
      } catch (err) {
        console.error("Error fetching analyses:", err);
        const errorMessage =
          err.response?.data?.detail || "Failed to fetch analyses";
        setError(errorMessage);
        onError(new Error(errorMessage));
      } finally {
        setLoading(false);
      }
    };
    fetchAnalyses();
  }, []);

  const handleDelete = async (id) => {
    setLoading(true);
    console.log(`Deleting analysis with ID: ${id}`);
    try {
      await axios.delete(`http://localhost:8000/analysis/${id}`);
      setAnalyses(analyses.filter((analysis) => analysis.id !== id));
    } catch (err) {
      console.error("Error deleting analysis:", err);
      const errorMessage =
        err.response?.data?.detail || "Failed to delete analysis";
      setError(errorMessage);
      onError(new Error(errorMessage));
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (analysis) => {
    console.log("Editing analysis:", analysis);
    setEditingId(analysis.id);
    setEditText(analysis.text);
  };

  const handleUpdate = async (id) => {
    setLoading(true);
    console.log(`Updating analysis with ID: ${id}, new text: ${editText}`);
    try {
      await axios.put(`http://localhost:8000/analysis/${id}`, {
        text: editText,
      });
      setAnalyses(
        analyses.map((analysis) =>
          analysis.id === id ? { ...analysis, text: editText } : analysis
        )
      );
      setEditingId(null);
      setEditText("");
    } catch (err) {
      console.error("Error updating analysis:", err);
      const errorMessage =
        err.response?.data?.detail || "Failed to update analysis";
      setError(errorMessage);
      onError(new Error(errorMessage));
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "20px", color: "#666" }}>
        Loading analyses...
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          color: "red",
          padding: "20px",
          backgroundColor: "#ffebee",
          borderRadius: "5px",
          textAlign: "center",
        }}
      >
        {error}
      </div>
    );
  }

  return (
    <div
      style={{
        padding: "20px",
        backgroundColor: "#fff",
        borderRadius: "10px",
        width: "100%",
        marginTop: "20px",
        boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
      }}
    >
      <h3 style={{ fontWeight: "bold", textAlign: "center" }}>Past Analyses</h3>
      <ul style={{ listStyleType: "none", padding: "0" }}>
        {analyses.length === 0 ? (
          <p style={{ textAlign: "center", color: "#666" }}>
            No analyses available.
          </p>
        ) : (
          analyses.map((analysis) => (
            <li
              key={analysis.id}
              style={{
                marginBottom: "10px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              {editingId === analysis.id ? (
                <>
                  <input
                    type="text"
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    style={{
                      flex: 1,
                      marginRight: "10px",
                      padding: "5px",
                      borderRadius: "5px",
                      border: "1px solid #ccc",
                    }}
                    disabled={loading}
                  />
                  <button
                    onClick={() => handleUpdate(analysis.id)}
                    style={{
                      marginRight: "10px",
                      padding: "5px 10px",
                      backgroundColor: "#4CAF50",
                      color: "#fff",
                      border: "none",
                      borderRadius: "5px",
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.6 : 1,
                    }}
                    disabled={loading}
                  >
                    {loading ? "Saving..." : "Save"}
                  </button>
                  <button
                    onClick={() => setEditingId(null)}
                    style={{
                      padding: "5px 10px",
                      backgroundColor: "#ccc",
                      color: "#fff",
                      border: "none",
                      borderRadius: "5px",
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.6 : 1,
                    }}
                    disabled={loading}
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <span
                    onClick={() => onSelectAnalysis(analysis)}
                    style={{ cursor: "pointer", flex: 1, padding: "5px" }}
                  >
                    {analysis.text} - Adorescore:{" "}
                    {analysis.analysis_result.adorescore.overall}
                  </span>
                  <button
                    onClick={() => handleEdit(analysis)}
                    style={{
                      marginRight: "10px",
                      padding: "5px 10px",
                      backgroundColor: "#FFC107",
                      color: "#fff",
                      border: "none",
                      borderRadius: "5px",
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.6 : 1,
                    }}
                    disabled={loading}
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDelete(analysis.id)}
                    style={{
                      padding: "5px 10px",
                      backgroundColor: "#F44336",
                      color: "#fff",
                      border: "none",
                      borderRadius: "5px",
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.6 : 1,
                    }}
                    disabled={loading}
                  >
                    Delete
                  </button>
                </>
              )}
            </li>
          ))
        )}
      </ul>
    </div>
  );
};

export default AnalysisList;
