import React from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";

const EmotionRadarChart = ({ emotions, activation }) => {
  const data = emotions.map((emotion) => ({
    emotion: emotion.emotion,
    probability: emotion.probability * 100,
  }));

  return (
    <div
      style={{
        width: "100%",
        height: 300,
        backgroundColor: "#fff",
        borderRadius: "10px",
        padding: "10px",
        boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
      }}
    >
      <h3 style={{ textAlign: "center", fontWeight: "bold" }}>
        {activation} ACTIVATION EMOTIONS{" "}
        {data.length > 0 ? data[0].emotion : "Neutral"}
      </h3>
      <ResponsiveContainer>
        <RadarChart data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="emotion" />
          <PolarRadiusAxis angle={30} domain={[0, 100]} />
          <Radar
            name={activation}
            dataKey="probability"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.6}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default EmotionRadarChart;
