import os
import torch
import uvicorn
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import logging
from datetime import datetime, timedelta

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMOTION_THRESHOLD = 0.5
TOPIC_RELEVANCE_THRESHOLD = 0.3
MAX_TEXT_LENGTH = 512  # Based on typical transformer model limits
DEFAULT_RELEVANCE = 0.1

app = FastAPI(title="Text Analysis API", description="API for emotion, topic, and adorescore analysis")

# Load NLP resources
nlp = spacy.load("en_core_web_sm")

def get_db_connection():
    """Create and initialize SQLite database connection."""
    conn = sqlite3.connect('emotion_analysis.db')
    conn.row_factory = sqlite3.Row
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_data'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE analysis_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    analysis_result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        cursor.execute("PRAGMA table_info(analysis_data)")
        if 'created_at' not in {row[1] for row in cursor.fetchall()}:
            cursor.execute('ALTER TABLE analysis_data ADD COLUMN created_at TIMESTAMP')
            cursor.execute('UPDATE analysis_data SET created_at = ? WHERE created_at IS NULL', (datetime.now().isoformat(),))
    return conn

class AdvancedEmotionAnalyzer:
    def __init__(self, model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion", threshold: float = EMOTION_THRESHOLD):
        """Initialize the emotion analyzer with a specified model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.threshold = threshold
        self.emotion_labels = list(self.model.config.id2label.values())  # Dynamically fetch labels

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze emotions in the given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        emotions = [
            {"emotion": self.emotion_labels[i].lower(), "probability": float(prob)}
            for i, prob in enumerate(probs) if prob >= self.threshold
        ]
        if not emotions:
            max_idx = np.argmax(probs)
            emotions = [{"emotion": self.emotion_labels[max_idx].lower(), "probability": float(probs[max_idx])}]
        return {"text": text, "emotions": emotions}

class EnhancedTopicAnalyzer:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = nlp

    def extract_topics_and_subtopics(self, text: str, sim_threshold: float = 0.7) -> Tuple[List[str], Dict[str, List[str]]]:
        """Extract main topics and subtopics from text."""
        doc = self.nlp(text)
        candidate_topics = list({self._normalize_topic(ent.text) for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG", "GPE", "NORP", "EVENT"]})
        candidate_topics.extend(self._normalize_topic(chunk.text) for chunk in doc.noun_chunks if self._normalize_topic(chunk.text) not in candidate_topics)
        if not candidate_topics:
            return ["general"], {"general": []}
        embeddings = self.encoder.encode(candidate_topics)
        sim_matrix = cosine_similarity(embeddings)
        main_topics = []
        seen = set()
        for i in range(len(candidate_topics)):
            if i not in seen:
                main_topics.append(candidate_topics[i])
                seen.update(j for j in range(i + 1, len(candidate_topics)) if sim_matrix[i][j] >= sim_threshold)
        subtopics = self._extract_subtopics(doc, main_topics)
        return main_topics, subtopics

    def _normalize_topic(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        return " ".join(tokens) or text.lower()

    def _extract_subtopics(self, doc, main_topics: List[str]) -> Dict[str, List[str]]:
        subtopics = {topic: set() for topic in main_topics}
        topic_words = {topic: set(t.lemma_.lower() for t in self.nlp(topic)) or {topic.lower()} for topic in main_topics}
        for sent in doc.sents:
            for i, token in enumerate(sent):
                if token.pos_ in ["ADJ", "VERB", "NOUN"] and token.text.lower() not in main_topics:
                    phrase = [token.text.lower()]
                    if i + 1 < len(sent) and sent[i + 1].pos_ in ["NOUN", "ADJ", "VERB"]:
                        phrase.append(sent[i + 1].text.lower())
                    sub_text = " ".join(phrase)
                    closest_topic = min(
                        main_topics,
                        key=lambda t: min(abs(token.i - idx) for idx in [t.i for t in sent if t.lemma_.lower() in topic_words[t]]) or 100,
                        default=None
                    )
                    if closest_topic:
                        subtopics[closest_topic].add(sub_text)
        return {topic: list(subs) for topic, subs in subtopics.items()}

    def get_topic_relevance(self, text: str, topics: List[str]) -> Dict[str, float]:
        """Calculate relevance of topics to the text."""
        doc_embedding = self.encoder.encode([text])
        topic_embeddings = self.encoder.encode(topics)
        similarities = cosine_similarity(doc_embedding, topic_embeddings)[0]
        return {topic: float(sim) for topic, sim in zip(topics, similarities)}

class RefinedAdorescoreCalculator:
    def __init__(self):
        self.emotion_valence = {
            "joy": 0.8, "love": 0.8, "surprise": 0.5, "anger": -0.5, "fear": -0.6, "sadness": -0.4
        }

    def _is_topic_mentioned(self, sentence: str, topic: str, subtopics: List[str]) -> bool:
        keywords = [topic.lower()] + [st.lower() for st in subtopics]
        return any(kw in sentence.lower() for kw in keywords)

    def calculate_adorescore(self, sentence_emotions: List[List[Dict[str, Any]]], topics: List[str], subtopics: Dict[str, List[str]], relevance_scores: Dict[str, float], sentences: List[str]) -> Dict[str, Any]:
        """Calculate adorescore based on emotions and topic relevance."""
        topic_sentiments = {topic: [] for topic in topics}
        for sent, emotions in zip(sentences, sentence_emotions):
            sentiment_score = sum(e["probability"] * self.emotion_valence.get(e["emotion"], 0) for e in emotions)
            adjusted_score = (sentiment_score + 1) / 2 * 100  # Map to 0-100
            mentioned_topics = [topic for topic in topics if self._is_topic_mentioned(sent, topic, subtopics.get(topic, []))]
            for topic in mentioned_topics:
                relevance = relevance_scores.get(topic, DEFAULT_RELEVANCE)
                topic_sentiments[topic].append(adjusted_score * relevance)
        
        breakdown = {topic: int(np.mean(scores)) if scores else 50 for topic, scores in topic_sentiments.items()}
        total_relevance = sum(relevance_scores.values()) or 1
        overall = int(sum(breakdown[t] * relevance_scores.get(t, DEFAULT_RELEVANCE) for t in breakdown) / total_relevance)
        return {"overall": overall, "breakdown": breakdown}

class IntegratedAnalyzer:
    def __init__(self):
        self.emotion_analyzer = AdvancedEmotionAnalyzer()
        self.topic_analyzer = EnhancedTopicAnalyzer()
        self.score_calculator = RefinedAdorescoreCalculator()

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform integrated text analysis."""
        overall_emotions = self.emotion_analyzer.analyze_text(text)["emotions"]
        main_topics, subtopics = self.topic_analyzer.extract_topics_and_subtopics(text)
        relevance_scores = self.topic_analyzer.get_topic_relevance(text, main_topics)
        filtered_topics = [t for t in main_topics if relevance_scores.get(t, 0) >= TOPIC_RELEVANCE_THRESHOLD]
        filtered_subtopics = {t: subtopics.get(t, []) for t in filtered_topics}
        filtered_relevance = {t: max(DEFAULT_RELEVANCE, relevance_scores[t]) for t in filtered_topics}
        
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        sentence_emotions = [self.emotion_analyzer.analyze_text(sent)["emotions"] for sent in sentences]
        adorescore = self.score_calculator.calculate_adorescore(sentence_emotions, filtered_topics, filtered_subtopics, filtered_relevance, sentences)
        
        return {
            "emotion_analysis": {"text": text, "emotions": overall_emotions},
            "topics": {"main": filtered_topics, "subtopics": filtered_subtopics, "relevance": filtered_relevance},
            "adorescore": adorescore
        }

analyzer = IntegratedAnalyzer()

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, description="Text to analyze")

class AnalysisResponse(BaseModel):
    id: int
    text: str
    analysis_result: Dict[str, Any]

def create_analysis(text: str, analysis_result: Dict[str, Any]) -> int:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO analysis_data (text, analysis_result) VALUES (?, ?)', (text, json.dumps(analysis_result)))
        conn.commit()
        return cursor.lastrowid

def read_analysis(analysis_id: int = None, last_30_days: bool = False, limit: int = 10, offset: int = 0) -> Any:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if analysis_id:
            cursor.execute('SELECT * FROM analysis_data WHERE id = ?', (analysis_id,))
            return cursor.fetchone()
        query = 'SELECT * FROM analysis_data'
        params = []
        if last_30_days:
            query += ' WHERE created_at >= ?'
            params.append((datetime.now() - timedelta(days=30)).isoformat())
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        cursor.execute(query, params)
        return cursor.fetchall()

def update_analysis(analysis_id: int, text: str = None, analysis_result: Dict[str, Any] = None) -> bool:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        updates, params = [], []
        if text:
            updates.append("text = ?")
            params.append(text)
        if analysis_result:
            updates.append("analysis_result = ?")
            params.append(json.dumps(analysis_result))
        if not updates:
            return False
        params.append(analysis_id)
        cursor.execute(f'UPDATE analysis_data SET {", ".join(updates)} WHERE id = ?', params)
        conn.commit()
        return cursor.rowcount > 0

def delete_analysis(analysis_id: int) -> bool:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM analysis_data WHERE id = ?', (analysis_id,))
        conn.commit()
        return cursor.rowcount > 0

@app.get("/analyses", response_model=List[AnalysisResponse])
async def get_all_analyses(last_30_days: bool = False, limit: int = 10, offset: int = 0):
    """Retrieve all analyses with optional filtering and pagination."""
    try:
        results = read_analysis(last_30_days=last_30_days, limit=limit, offset=offset)
        formatted_results = [
            {"id": row["id"], "text": row["text"], "analysis_result": json.loads(row["analysis_result"] or "{}")}
            for row in results if row
        ]
        logger.info(f"Fetched {len(formatted_results)} analyses")
        return formatted_results
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.error(f"Error fetching analyses: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analyses")

@app.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: int):
    """Retrieve a specific analysis by ID."""
    result = read_analysis(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"id": result["id"], "text": result["text"], "analysis_result": json.loads(result["analysis_result"] or "{}")}

@app.put("/analysis/{analysis_id}")
async def update_analysis_endpoint(analysis_id: int, request: TextRequest | None = None, analysis_result: Dict[str, Any] | None = None):
    """Update an existing analysis."""
    if not request and not analysis_result:
        raise HTTPException(status_code=400, detail="No update data provided")
    text = request.text if request else None
    if not update_analysis(analysis_id, text, analysis_result):
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"message": "Analysis updated successfully"}

@app.delete("/analysis/{analysis_id}")
async def delete_analysis_endpoint(analysis_id: int):
    """Delete an analysis by ID."""
    if not delete_analysis(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"message": "Analysis deleted successfully"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text_endpoint(request: TextRequest):
    """Analyze text for emotions, topics, and adorescore."""
    try:
        logger.info(f"Analyzing text: {request.text[:50]}...")
        result = analyzer.analyze_text(request.text)
        analysis_id = create_analysis(request.text, result)
        return {"id": analysis_id, "text": request.text, "analysis_result": result}
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze text")

# CORS configuration
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)