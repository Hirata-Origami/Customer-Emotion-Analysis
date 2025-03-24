import React, { useState, useEffect } from "react";
import styled from "styled-components";
import EmotionRadarChart from "./components/EmotionRadarChart";
import AdorescoreDisplay from "./components/AdorescoreDisplay";
import TopThemesList from "./components/TopThemesList";
import AnalysisForm from "./components/AnalysisForm";
import AnalysisList from "./components/AnalysisList";
import TopicHierarchy from "./components/TopicHierarchy";

const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  background-color: #f0f0f0;
  min-height: 100vh;
`;

const ChartsContainer = styled.div`
  display: flex;
  justify-content: space-between;
  width: 100%;
  margin-top: 20px;
  gap: 20px;
  flex-wrap: wrap;
`;

const ResultsContainer = styled.div`
  width: 100%;
  margin-top: 20px;
`;

const Loading = styled.div`
  text-align: center;
  padding: 20px;
  color: #666;
`;

const Error = styled.div`
  color: red;
  padding: 20px;
  background-color: #ffebee;
  border: 1px solid #ffcdd2;
  border-radius: 5px;
  text-align: center;
`;

const App = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalysisComplete = (data) => {
    console.log("Analysis completed:", data);
    if (!data || !data.analysis_result) {
      console.error("Invalid analysis data received:", data);
      setError("Invalid analysis data received from backend");
      setAnalysisData(null);
      return;
    }
    setLoading(false);
    setAnalysisData(data.analysis_result);
    console.log("Parsed analysisData for debugging:", data.analysis_result);
    setError(null);
  };

  const handleSelectAnalysis = (analysis) => {
    console.log("Selected analysis:", analysis);
    if (!analysis || !analysis.analysis_result) {
      console.error("Invalid analysis selected:", analysis);
      setError("Invalid analysis data selected");
      setAnalysisData(null);
      return;
    }
    setAnalysisData(analysis.analysis_result);
    console.log(
      "Selected analysisData for debugging:",
      analysis.analysis_result
    ); // Add logging
    setError(null);
  };

  const handleError = (err) => {
    console.error("Error in App:", err);
    setLoading(false);
    setError(err.message || "An error occurred");
    setAnalysisData(null);
  };

  useEffect(() => {
    console.log("App mounted or updated. Current analysisData:", analysisData);
    if (!analysisData && !loading && !error) {
      fetchInitialAnalysis();
    }
  }, [analysisData, loading, error]);

  const fetchInitialAnalysis = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/analyses");
      if (!response.ok) throw new Error("Failed to fetch initial analysis");
      const data = await response.json();
      console.log("Initial analyses fetched:", data);
      if (data.length > 0 && data[0].analysis_result) {
        setAnalysisData(data[0].analysis_result);
      } else {
        console.warn("No valid analysis data available in initial fetch");
        setAnalysisData(null);
      }
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <Loading>Loading...</Loading>;
  }

  if (error) {
    return <Error>{error}</Error>;
  }

  if (
    !analysisData ||
    !analysisData.emotion_analysis ||
    !analysisData.emotion_analysis.emotions
  ) {
    return (
      <div style={{ textAlign: "center", padding: "20px", color: "#666" }}>
        No analysis data available. Please enter text to analyze or select a
        past analysis.
      </div>
    );
  }

  const themes = Object.entries(analysisData.adorescore.breakdown || {}).map(
    ([name, score]) => ({
      name,
      score,
      volume: analysisData.topics?.relevance?.[name]
        ? (analysisData.topics.relevance[name] * 100).toFixed(0)
        : 0,
    })
  );

  const highActivationEmotions = analysisData.emotion_analysis.emotions.filter(
    (e) => e.probability >= 0.7
  );
  const mediumActivationEmotions =
    analysisData.emotion_analysis.emotions.filter(
      (e) => e.probability >= 0.4 && e.probability < 0.7
    );
  const lowActivationEmotions = analysisData.emotion_analysis.emotions.filter(
    (e) => e.probability < 0.4
  );

  return (
    <AppContainer>
      <h1 style={{ fontWeight: "bold", marginBottom: "20px" }}>
        Emotion Analysis Dashboard
      </h1>
      <AnalysisForm
        onAnalysisComplete={handleAnalysisComplete}
        onError={handleError}
      />
      <ChartsContainer>
        <EmotionRadarChart
          emotions={
            highActivationEmotions.length > 0
              ? highActivationEmotions
              : analysisData.emotion_analysis.emotions
          }
          activation="High"
        />
        <EmotionRadarChart
          emotions={
            mediumActivationEmotions.length > 0
              ? mediumActivationEmotions
              : analysisData.emotion_analysis.emotions
          }
          activation="Medium"
        />
        <EmotionRadarChart
          emotions={
            lowActivationEmotions.length > 0
              ? lowActivationEmotions
              : analysisData.emotion_analysis.emotions
          }
          activation="Low"
        />
        <AdorescoreDisplay
          adorescore={analysisData.adorescore.overall}
          primaryEmotion={
            analysisData.emotion_analysis.primary_emotion || {
              emotion: "Joy",
              probability: 0.8,
              confidence: 0.95,
            }
          }
        />
      </ChartsContainer>
      <ResultsContainer>
        <TopThemesList themes={themes} />
        <TopicHierarchy analysisData={analysisData} />
      </ResultsContainer>
      <AnalysisList
        onSelectAnalysis={handleSelectAnalysis}
        onError={handleError}
      />
    </AppContainer>
  );
};

export default App;
