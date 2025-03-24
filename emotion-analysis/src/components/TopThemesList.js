import React from "react";

const TopThemesList = ({ themes }) => {
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
      <h3 style={{ fontWeight: "bold", textAlign: "center" }}>
        Top Themes in Dataset
      </h3>
      <ul style={{ listStyleType: "disc", paddingLeft: "20px" }}>
        {themes.map((theme, index) => (
          <li key={index}>
            {theme.name} - Score: {theme.score} - Volume: {theme.volume}%
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TopThemesList;
