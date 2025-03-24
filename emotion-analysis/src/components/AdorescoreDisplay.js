import React from "react";

const AdorescoreDisplay = ({ adorescore, primaryEmotion }) => {
  return (
    <div
      style={{
        textAlign: "center",
        padding: "20px",
        backgroundColor: "#fff",
        borderRadius: "10px",
        width: "100%",
        boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
      }}
    >
      <h2 style={{ fontWeight: "bold" }}>Adorescore: {adorescore}</h2>
      <p>
        Driven by {primaryEmotion.emotion} - {primaryEmotion.probability * 100}%
      </p>
    </div>
  );
};

export default AdorescoreDisplay;
