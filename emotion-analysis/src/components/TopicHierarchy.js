import React from "react";
import styled from "styled-components";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const TopicContainer = styled.div`
  padding: 20px;
  background-color: #fff;
  border-radius: 10px;
  width: 100%;
  margin-top: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h3`
  font-weight: bold;
  text-align: center;
  color: #343a40;
`;

const TopicItem = styled.div`
  margin: 10px 0;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 5px;
`;

const SubtopicList = styled.ul`
  list-style-type: disc;
  padding-left: 20px;
  margin: 10px 0;
`;

const ScoreDisplay = styled.div`
  margin-top: 10px;
  font-size: 14px;
  color: #666;
`;

const NoDataMessage = styled.p`
  text-align: center;
  color: #666;
  margin: 10px 0;
`;

const TopicHierarchy = ({ analysisData }) => {
  if (!analysisData) {
    return (
      <TopicContainer>
        <Title>No analysis data available.</Title>
      </TopicContainer>
    );
  }

  const topics = analysisData.topics?.main || [];
  const subtopics = analysisData.topics?.subtopics || {};
  const adorescoreBreakdown = analysisData.adorescore?.breakdown || {};
  const adorescoreOverall = analysisData.adorescore?.overall || 0;

  if (topics.length === 0 && Object.keys(adorescoreBreakdown).length === 0) {
    return (
      <TopicContainer>
        <Title>No topics or Adorescore breakdown available.</Title>
        <NoDataMessage>
          The input text may be neutral or lacks significant topics. Adorescore:{" "}
          {adorescoreOverall}
        </NoDataMessage>
      </TopicContainer>
    );
  }

  const chartData = topics.map((topic) => ({
    name: topic,
    score: adorescoreBreakdown[topic] || 0,
  }));

  return (
    <TopicContainer>
      <Title>Topic Hierarchy and Adorescores</Title>
      {topics.length > 0 ? (
        topics.map((topic, index) => (
          <TopicItem key={index}>
            <h4 style={{ margin: 0, color: "#343a40" }}>{topic}</h4>
            <SubtopicList>
              {(subtopics[topic] || []).map((subtopic, subIndex) => (
                <li key={subIndex}>{subtopic}</li>
              )) || <li>No subtopics available.</li>}
            </SubtopicList>
            <ScoreDisplay>
              Adorescore for {topic}: {adorescoreBreakdown[topic] || 0}
            </ScoreDisplay>
          </TopicItem>
        ))
      ) : (
        <NoDataMessage>No topics detected in the analysis.</NoDataMessage>
      )}
      {Object.keys(adorescoreBreakdown).length > 0 && chartData.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ textAlign: "center", color: "#343a40" }}>
            Adorescore Breakdown
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="score" fill="#007bff" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </TopicContainer>
  );
};

export default TopicHierarchy;
