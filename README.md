# Emotion Analysis System Documentation

**Version:** 1.0  
**Date:** February 25, 2025  
**Author:** Grok 3 (xAI)

## Overview

This document provides an in-depth overview of the Emotion Analysis System, a full-stack application for analyzing text emotions, topics, and Adorescores. It consists of a React frontend and a FastAPI backend with SQLite persistence, designed for users to input text, visualize results, and manage analyses. The system leverages NLP technologies (spaCy, transformers, sentence-transformers, VADER) to process text and calculate scores dynamically.

## System Architecture

### Frontend (React)

- **Built with:** React 18 (or 19), styled-components, Recharts, and Axios.
- **Runs on:** `http://localhost:3000`
- **Components:**
  - `App`
  - `AnalysisForm`
  - `AnalysisList`
  - `EmotionRadarChart`
  - `AdorescoreDisplay`
  - `TopThemesList`
  - `TopicHierarchy`

### Backend (FastAPI)

- **Built with:** Python 3.8+, FastAPI, uvicorn, and NLP libraries.
- **Runs on:** `http://localhost:8000`
- **Modules:**
  - `AdvancedEmotionAnalyzer`
  - `EnhancedTopicAnalyzer`
  - `RefinedAdorescoreCalculator`
  - `IntegratedAnalyzer`
- **Database:** SQLite for persistence, storing analysis data in `emotion_analysis.db` with the `analysis_data` table (id, text, analysis_result, created_at).

### Data Flow

1. User inputs text in the frontend → POST to `/analyze`.
2. Backend processes text (emotions via DistilBERT, topics via spaCy/SentenceTransformer, Adorescores via sentiment/valence) → Stores in SQLite.
3. Returns results to frontend → Displays in charts, lists, and scores.
4. Users manage past analyses via GET/PUT/DELETE endpoints.

### Communication

- **Frontend communicates with backend via:** RESTful APIs (CORS-enabled).
- **Data exchange format:** JSON, ensuring compatibility between React and FastAPI.

## System Requirements

### Hardware

- Minimal (modern laptop/desktop with 8GB RAM, 2GHz CPU).

### Software

- **Frontend:** Node.js 16+/npm 8+, modern browser.
- **Backend:** Python 3.8+/pip, SQLite (included with Python).
- **Dependencies:** See individual frontend/backend docs.

## Installation and Setup

### Frontend Setup

```bash
cd frontend
npm install react@18.2.0 react-dom@18.2.0 styled-components axios recharts
npm start
```

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py
```

### Run Both

- Frontend on `http://localhost:3000`
- Backend on `http://localhost:8000`
- Ensure both are running simultaneously for full functionality.

## Functionality

### Text Analysis

- Users input text, receiving emotions (high/medium/low activation), topics, subtopics, and Adorescores (40–80 range).

### Visualization

- Radar charts for emotions, bar charts for Adorescores, lists for themes/topics/subtopics.

### Past Analyses

- View, edit, or delete historical analyses, stored in SQLite.

### Error Handling

- Robust error boundaries, logging, and HTTP exceptions for reliability.

## Key Features

- Real-time NLP processing with modern models.
- Persistent data storage for historical tracking.
- Responsive, interactive UI with modern visualizations.
- Scalable architecture for future enhancements.

## Usage

### Access the Dashboard

- Open `http://localhost:3000` in a browser after starting both frontend and backend.

### Analyze Text

- Enter text (e.g., "Delivery fast") in the form, click "Analyze," and view results in charts and lists.

### Manage Analyses

- View past analyses, click to select one, edit text, or delete entries via the list.

### Explore Visualizations

- Hover over charts for details, scroll through lists for themes/topics.

## Maintenance and Troubleshooting

### Common Issues

- **Connection Errors:** Ensure both frontend and backend are running, and ports (3000, 8000) are open. Check CORS settings.
- **Dependency Conflicts:** Use `npm audit fix` or `pip install --upgrade <package>` to resolve issues.
- **Model Loading:** Verify spaCy and transformers models are downloaded and accessible.

### Updates

- Regularly update React, FastAPI, and NLP libraries to maintain compatibility.
- Sync frontend/backend changes to preserve API contracts.

### Testing

- **Frontend:** Use Jest/React Testing Library for components.
- **Backend:** Use pytest for API and logic tests.
- **System:** Test end-to-end with sample texts, verifying scores, visualizations, and persistence.

## Performance and Scalability

### Current Performance

- Handles single-text analysis efficiently on modest hardware, with SQLite suitable for small datasets.

### Scalability

- Use a production database (e.g., PostgreSQL) for larger datasets.
- Implement caching (e.g., Redis) for frequent analyses.
- Scale backend with uvicorn workers or deploy on cloud platforms.

## Security

### CORS

- Restricted to `http://localhost:3000` for development; update for production.

### Data

- SQLite stores data locally; encrypt sensitive data or use a secure database for production.

### Input Validation

- FastAPI’s Pydantic models validate text inputs, preventing injection attacks.

## Future Enhancements

- Add user authentication for multi-user support.
- Implement real-time collaboration or batch processing.
- Enhance NLP with additional models (e.g., BERT, RoBERTa) for better accuracy.
- Improve UI with animations, dark mode, or multilingual support.
