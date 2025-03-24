# Python (FastAPI) Backend Documentation

## Emotion Analysis Backend Documentation

**Version:** 1.0  
**Date:** February 25, 2025  
**Author:** Grok 3 (xAI)

## Overview

This document describes the FastAPI backend for the Emotion Analysis Dashboard, a Python application that processes text inputs, analyzes emotions, topics, and Adorescores, and stores results in a SQLite database (`emotion_analysis.db`). It uses NLP libraries (spaCy, transformers, sentence-transformers, VADER) and provides RESTful APIs for frontend interaction.

## System Requirements

- **Python:** Version 3.8 or higher (recommended 3.11)
- **Pip:** For dependency management

### Dependencies

- `fastapi` (0.100.x)
- `uvicorn` (0.22.x)
- `torch` (2.x)
- `transformers` (4.x)
- `sentence-transformers` (2.x)
- `scikit-learn` (1.x)
- `spacy` (3.x, with `en_core_web_sm`)
- `vaderSentiment` (3.x)
- `numpy` (1.x)
- `sqlite3` (included with Python)

## Installation and Setup

### Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

### Download the spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Run the Application

Start the server:

```bash
python main.py
```

The app runs on [http://localhost:8000](http://localhost:8000).

## Database

A SQLite database (`emotion_analysis.db`) is created automatically in the project directory, storing analysis data in the `analysis_data` table.

## Architecture and Modules

The backend is structured as a single `main.py` file with the following components:

- **FastAPI App:** Defines endpoints for text analysis and analysis management, with CORS middleware for frontend compatibility.
- **Analyzers:**
  - **AdvancedEmotionAnalyzer:** Uses DistilBERT for emotion classification, returning probabilities and confidences.
  - **EnhancedTopicAnalyzer:** Extracts topics and subtopics using SentenceTransformer and spaCy, calculating relevance scores.
  - **RefinedAdorescoreCalculator:** Computes Adorescores (40–80 range) based on emotions, sentiments, and topics.
  - **IntegratedAnalyzer:** Combines the above for full text analysis.
- **Database Operations:** Uses SQLite for CRUD operations on `analysis_data` (id, text, analysis_result, created_at).
- **Logging:** Implements logging for debugging and error tracking.

### Data Flow

Text input → IntegratedAnalyzer → Emotion, Topic, and Adorescore analysis → Store in SQLite → Return results via API.

## Functionality

### Text Analysis (`/analyze`)

Accepts POST requests with text, analyzes emotions (via DistilBERT), topics/subtopics (via spaCy/SentenceTransformer), and Adorescores (via sentiment and valence), storing results in the database.

### Past Analyses Management (`/analyses`, `/analysis/{id}`)

- **GET /analyses:** Retrieves all analyses (optionally last 30 days, paginated).
- **GET /analysis/{id}:** Fetches a specific analysis.
- **PUT /analysis/{id}:** Updates text or results.
- **DELETE /analysis/{id}:** Deletes an analysis.

### Scoring

Adorescores are calculated dynamically (40–80 range) based on emotion valence, VADER sentiment, and topic relevance, handling negations and contrasts ("but," "nevertheless").

### Subtopic Extraction

Extracts descriptive subtopics (e.g., "fast," "amazing") while excluding topics (e.g., "delivery," "quality") as subtopics.

## Key Features

- Real-time text processing with NLP models.
- Persistent storage in SQLite.
- RESTful API with CORS for React integration.
- Error handling with HTTP exceptions and logging.

## Usage

### Launching the Backend

Follow installation steps, navigate to the backend directory, and run `python main.py`.

### Testing Endpoints

Use curl, Postman, or the frontend to test APIs:

#### POST /analyze

```bash
curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d '{"text": "Delivery fast"}'
```

Returns JSON with id, text, and analysis_result (emotions, topics, Adorescore).

#### GET /analyses

```bash
curl "http://localhost:8000/analyses"
```

Returns a list of past analyses.

#### GET /analysis/1

```bash
curl "http://localhost:8000/analysis/1"
```

Retrieves analysis with id=1.

#### PUT /analysis/1

```bash
curl -X PUT "http://localhost:8000/analysis/1" -H "Content-Type: application/json" -d '{"text": "Updated text"}'
```

Updates the analysis.

#### DELETE /analysis/1

```bash
curl -X DELETE "http://localhost:8000/analysis/1"
```

Deletes the analysis.

### Database Inspection

Use `sqlite3 emotion_analysis.db` to query `analysis_data` and verify stored results.

## Maintenance and Troubleshooting

### Common Issues

- **Model Loading Errors:** Ensure spaCy (`en_core_web_sm`) and transformers models are downloaded. Check pip install logs.
- **Database Errors:** Verify `emotion_analysis.db` is writable and SQLite is accessible. Check permissions or path issues.
- **API Errors:** Check logs for HTTPException details (e.g., 500 for analysis failures, 404 for missing analyses).

### Updates

Update Python libraries regularly via `pip install --upgrade <package>`. Sync with frontend changes to maintain API compatibility.

### Testing

Use pytest for unit tests:

```bash
pip install pytest
pytest main.py -v
```

Test endpoints with mock data to verify analysis and database operations.

## API Reference

### /analyze (POST)

- **Request:** `{"text": "string"}`
- **Response:** `{"id": int, "text": "string", "analysis_result": {"emotion_analysis": {...}, "topics": {...}, "adorescore": {...}}}`
- **Status:** 200 OK, 500 Internal Server Error

### /analyses (GET)

- **Query Params:** `last_30_days: bool, limit: int, offset: int`
- **Response:** List of `{"id": int, "text": "string", "analysis_result": {...}}`
- **Status:** 200 OK, 500 Internal Server Error

### /analysis/{id} (GET, PUT, DELETE)

- **GET:** Returns specific analysis (200 OK, 404 Not Found)
- **PUT:** Updates text/results (`{"text": "string"}` or `{"analysis_result": {...}}`, 200 OK, 404 Not Found, 400 Bad Request)
- **DELETE:** Deletes analysis (200 OK, 404 Not Found)

## Future Enhancements

- Add batch processing for multiple texts in `/analyze`.
- Implement caching for frequent analyses to improve performance.
- Enhance topic/subtopic extraction with additional NLP models (e.g., BERTopic).
