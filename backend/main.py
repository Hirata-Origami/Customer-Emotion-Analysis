import torch
import uvicorn
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
from typing import List, Dict, Any, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

nlp = spacy.load("en_core_web_sm")
vader_analyzer = SentimentIntensityAnalyzer()

def get_db_connection():
    conn = sqlite3.connect('emotion_analysis.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_data'")
    table_exists = cursor.fetchone()
    if not table_exists:
        cursor.execute('''
            CREATE TABLE analysis_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                analysis_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    else:
        cursor.execute("PRAGMA table_info(analysis_data)")
        columns = {row[1] for row in cursor.fetchall()}
        if 'created_at' not in columns:
            cursor.execute('''
                ALTER TABLE analysis_data ADD COLUMN created_at TIMESTAMP
            ''')
            current_time = datetime.now().isoformat()
            cursor.execute('''
                UPDATE analysis_data SET created_at = ? WHERE created_at IS NULL
            ''', (current_time,))
    conn.commit()
    return conn

class AdvancedEmotionAnalyzer:
    def __init__(self, model_name="joeddav/distilbert-base-uncased-go-emotions-student", threshold=0.5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.threshold = threshold
        self.emotion_labels = list(self.model.config.label2id.keys())
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        emotions = []
        for i, prob in enumerate(probs):
            if prob >= self.threshold:
                confidence = min(0.95, prob * 1.1)
                adjusted_prob = max(0.1, prob)
                emotions.append({"emotion": self.emotion_labels[i].lower(), "probability": float(adjusted_prob), "confidence": float(confidence)})
        if not emotions:
            emotions.append({"emotion": "neutral", "probability": 0.5, "confidence": 0.95})
        return {"text": text, "emotions": emotions}

class EnhancedTopicAnalyzer:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract_topics_and_subtopics(self, text: str, sim_threshold: float = 0.7) -> Tuple[List[str], Dict[str, List[str]]]:
        doc = self.nlp(text)
        candidate_topics = []
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "GPE", "NORP", "EVENT"]:
                candidate_topics.append(self._normalize_topic(ent.text))
        for chunk in doc.noun_chunks:
            normalized = self._normalize_topic(chunk.text)
            if normalized not in candidate_topics:
                candidate_topics.append(normalized)
        if not candidate_topics:
            return ["general"], {"general": []}
        embeddings = self.encoder.encode(candidate_topics)
        sim_matrix = cosine_similarity(embeddings)
        main_topics = []
        seen = set()
        for i in range(len(candidate_topics)):
            if i not in seen:
                main_topic = candidate_topics[i]
                main_topics.append(main_topic)
                for j in range(i + 1, len(candidate_topics)):
                    if sim_matrix[i][j] >= sim_threshold:
                        seen.add(j)
        subtopics = self._extract_subtopics(doc, main_topics)
        return main_topics, subtopics
    
    def _normalize_topic(self, text: str) -> str:
        doc_topic = self.nlp(text)
        tokens = []
        for token in doc_topic:
            if token.pos_ in ["NOUN", "PROPN"]:
                tokens.append(token.lemma_.lower())
            elif token.pos_ == "ADJ" and tokens and token.dep_ in ["amod"]:
                tokens.insert(0, token.text.lower())
        return " ".join(tokens) if tokens else text.lower()
    
    def _extract_subtopics(self, doc, main_topics):
        subtopics = {topic: set() for topic in main_topics}
        topic_words = {topic: set([t.lemma_.lower() for t in self.nlp(topic) if t.pos_ in ["NOUN", "PROPN"]]) or {topic.lower()} for topic in main_topics}
        for sent in doc.sents:
            topic_indices = {topic: [t.i for t in sent if t.lemma_.lower() in topic_words[topic]] for topic in main_topics}
            i = 0
            while i < len(sent):
                token = sent[i]
                phrase = []
                if token.pos_ in ["ADJ", "VERB", "NOUN"] and token.dep_ not in ["advmod", "cc", "mark"] and token.text.lower() not in [t.lower() for t in main_topics]:
                    phrase.append(token.text.lower())
                    i += 1
                    if i < len(sent) and sent[i].pos_ in ["NOUN", "ADJ", "VERB"] and sent[i].dep_ not in ["advmod", "cc", "mark"] and sent[i].text.lower() not in [t.lower() for t in main_topics]:
                        phrase.append(sent[i].text.lower())
                        i += 1
                else:
                    i += 1
                if phrase and " ".join(phrase) not in main_topics:
                    sub_text = " ".join(phrase)
                    head = token.head if token.i > 0 else token
                    self._assign_subtopic(head, sub_text, sent, subtopics, topic_words, topic_indices, main_topics)
        return {topic: list(candidates) for topic, candidates in subtopics.items()}
    
    def _is_negation(self, token, sent) -> bool:
        if token.dep_ == "neg" or token.text.lower() in ["not", "n't"]:
            return True
        if token.pos_ == "ADV" and token.lemma_.lower() in ["never", "no", "nor"]:
            return True
        for child in token.children:
            if child.dep_ == "neg" or child.text.lower() in ["not", "n't"]:
                return True
        for ancestor in token.ancestors:
            if ancestor.dep_ == "neg" or ancestor.text.lower() in ["not", "n't"]:
                return True
        return False
    
    def _assign_subtopic(self, head, sub_text, sent, subtopics, topic_words, topic_indices, main_topics):
        assigned = False
        for topic in main_topics:
            topic_lemmas = topic_words[topic]
            if (head.lemma_.lower() in topic_lemmas and head.dep_ in ["amod", "acomp", "advmod", "attr", "nsubj", "advcl", "neg"]) or (head.dep_ == "ROOT" and any(child.lemma_.lower() in topic_lemmas for child in head.children)) or (head.dep_ in ["dobj", "nsubj"] and head.pos_ in ["VERB", "AUX"] and any(t.lemma_.lower() in topic_lemmas for t in sent)):
                subtopics[topic].add(sub_text)
                assigned = True
                break
        if not assigned:
            distances = {topic: min([abs(head.i - idx) for idx in indices] or [100]) for topic, indices in topic_indices.items() if indices}
            if distances:
                closest_topic = min(distances, key=distances.get)
                if distances[closest_topic] <= 4:
                    subtopics[closest_topic].add(sub_text)
    
    def get_topic_relevance(self, text: str, topics: List[str]) -> Dict[str, float]:
        doc_embedding = self.encoder.encode([text])
        topic_embeddings = self.encoder.encode(topics)
        similarities = cosine_similarity(doc_embedding, topic_embeddings)[0]
        return {topic: float(sim) for topic, sim in zip(topics, similarities)}

class RefinedAdorescoreCalculator:
    def __init__(self):
        self.emotion_valence = {
            "joy": 0.8, "admiration": 0.7, "excitement": 0.7, "love": 0.8, "amusement": 0.6,
            "anger": -0.5, "disappointment": -0.6, "sadness": -0.4, "fear": -0.6, "annoyance": -0.5,
            "confusion": 0.0, "caring": 0.6, "curiosity": 0.5, "nervousness": -0.5, "pride": 0.7,
            "rage": -0.7, "vigilance": 0.5, "disgust": -0.5, "trust": 0.7, "anticipation": 0.5,
            "interest": 0.5, "acceptance": 0.6, "pensiveness": -0.4, "distraction": 0.0, "apprehension": -0.5,
            "serenity": 0.7, "ecstasy": 0.8, "neutral": 0.0
        }
    
    def _is_topic_mentioned(self, sentence, topic, subtopics):
        keywords = [topic.lower()] + [st.lower() for st in subtopics.get(topic, [])]
        return any(kw in sentence.lower() for kw in keywords)
    
    def calculate_adorescore(self, emotions_list: List[List[Dict[str, Any]]], topics: List[str], subtopics: Dict[str, List[str]], relevance_scores: Dict[str, float], sentiments: List[float], sentences: List[str], emotion_analyzer) -> Dict[str, Any]:
        all_emotions = emotions_list[0] if emotions_list else []
        if not all_emotions:
            max_positive = max(self.emotion_valence.items(), key=lambda x: x[1])[0]
            max_negative = min(self.emotion_valence.items(), key=lambda x: x[1])[0]
            primary = {"emotion": max_positive, "probability": 0.7, "confidence": 0.95}
            secondary = {"emotion": max_negative, "probability": 0.5, "confidence": 0.9}
        else:
            sorted_emotions = sorted(all_emotions, key=lambda x: x["probability"], reverse=True)
            primary = sorted_emotions[0]
            secondary = sorted_emotions[1] if len(sorted_emotions) > 1 else {"emotion": min(self.emotion_valence.items(), key=lambda x: x[1])[0], "probability": 0.5, "confidence": 0.9}
        
        before_split_text = " ".join(sentences)
        after_split_text = ""
        split_occurred = False
        if len(sentences) == 1:
            sent_doc = nlp(sentences[0])
            for token in sent_doc:
                if token.dep_ in ["cc", "mark"] or (token.pos_ == "ADV" and token.dep_ in ["advmod"]):
                    split_index = token.i
                    before_split_text = " ".join(t.text for t in sent_doc[:split_index])
                    after_split_text = " ".join(t.text for t in sent_doc[split_index + 1:])
                    split_occurred = True
                    break
        
        if split_occurred:
            before_split_result = emotion_analyzer.analyze_text(before_split_text)
            after_split_result = emotion_analyzer.analyze_text(after_split_text)
            primary_candidates = [(e["emotion"], e["probability"], e["confidence"], sentiments[0]) for e in before_split_result["emotions"]]
            primary = max([(e, p, c, s) for e, p, c, s in primary_candidates if self.emotion_valence.get(e, 0) > 0], key=lambda x: x[1], default=(max(self.emotion_valence.items(), key=lambda x: x[1])[0], 0.7, 0.95, 0))
            secondary_candidates = [(e["emotion"], e["probability"], e["confidence"], sentiments[0]) for e in after_split_result["emotions"]]
            primary_valence = self.emotion_valence.get(primary[0], 0)
            secondary = max([(e, p, c, s) for e, p, c, s in secondary_candidates if self.emotion_valence.get(e, 0) * primary_valence < 0], key=lambda x: abs(self.emotion_valence.get(x[0], 0)) * x[1], default=(min(self.emotion_valence.items(), key=lambda x: x[1])[0], 0.5, 0.9, 0))
        else:
            all_emotions = [(e["emotion"], e["probability"], e["confidence"], s) for e, s in zip(emotions_list[0], sentiments) if e]
            primary = max([(e, p, c, s) for e, p, c, s in all_emotions if self.emotion_valence.get(e, 0) > 0], key=lambda x: x[1], default=(max(self.emotion_valence.items(), key=lambda x: x[1])[0], 0.7, 0.95, 0))
            secondary = max([(e, p, c, s) for e, p, c, s in all_emotions if self.emotion_valence.get(e, 0) < 0], key=lambda x: abs(self.emotion_valence.get(x[0], 0)) * x[1], default=(min(self.emotion_valence.items(), key=lambda x: x[1])[0], 0.5, 0.9, 0))
        
        topic_sentiments = {topic: [] for topic in topics}
        for sent, score, emotions in zip(sentences, sentiments, emotions_list):
            sent_doc = nlp(sent)
            for topic in topic_sentiments.keys():
                if self._is_topic_mentioned(sent, topic, subtopics):
                    topic_words = [t.lower() for t in [topic] + subtopics.get(topic, [])]
                    primary_factor = (self.emotion_valence.get(primary[0], 0) * primary[1] + score) / 2 * 0.8
                    secondary_factor = (self.emotion_valence.get(secondary[0], 0) * secondary[1] + score) / 2 * 0.6
                    relevance = max(0.2, relevance_scores.get(topic, 0.1))  # Increase default relevance
                    adjusted_score = (primary_factor + secondary_factor) * relevance * 70 + 40
                    topic_sentiments[topic].append(max(40, min(80, adjusted_score)))
        
        breakdown = {topic: int(np.mean(scores)) if scores else 50 for topic, scores in topic_sentiments.items()}
        total_relevance = sum(relevance_scores.values()) or 1
        overall = int(sum(breakdown[t] * (relevance_scores.get(t, 0.1) + 0.1) for t in breakdown) / total_relevance)
        
        primary_emotion = {"emotion": primary[0], "probability": primary[1], "confidence": primary[2]}
        secondary_emotion = {"emotion": secondary[0], "probability": secondary[1], "confidence": secondary[2]}
        
        return {"emotion_analysis": {"text": " ".join(sentences), "emotions": emotions_list[0] if emotions_list else [], "primary_emotion": primary_emotion, "secondary_emotion": secondary_emotion}, "topics": {"main": topics, "subtopics": subtopics, "relevance": {t: max(0.1, r) for t, r in relevance_scores.items()}}, "adorescore": {"overall": overall, "breakdown": breakdown}}

class IntegratedAnalyzer:
    def __init__(self):
        self.emotion_analyzr = AdvancedEmotionAnalyzer()
        self.topic_analyzer = EnhancedTopicAnalyzer()
        self.score_calculator = RefinedAdorescoreCalculator()
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        main_topics, subtopics = self.topic_analyzer.extract_topics_and_subtopics(text)
        relevance_scores = self.topic_analyzer.get_topic_relevance(text, main_topics)
        filtered_topics = [t for t in main_topics if relevance_scores.get(t, 0) >= 0.3]
        filtered_subtopics = {t: subtopics.get(t, []) for t in filtered_topics}
        filtered_relevance = {t: max(0.1, relevance_scores[t]) for t in filtered_topics}
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        emotions_list = []
        sentiments = []
        for sent in sentences:
            emotion_result = self.emotion_analyzr.analyze_text(sent)
            emotions_list.append(emotion_result["emotions"])
            sentiments.append(vader_analyzer.polarity_scores(sent)["compound"])
        score_result = self.score_calculator.calculate_adorescore(emotions_list, filtered_topics, filtered_subtopics, filtered_relevance, sentiments, sentences, self.emotion_analyzr)
        return score_result

analyzer = IntegratedAnalyzer()

class TextRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    id: int
    text: str
    analysis_result: Dict[str, Any]

def create_analysis(text: str, analysis_result: Dict[str, Any]):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO analysis_data (text, analysis_result, created_at)
        VALUES (?, ?, ?)
    ''', (text, json.dumps(analysis_result), datetime.now()))
    conn.commit()
    analysis_id = cursor.lastrowid
    conn.close()
    return analysis_id

def read_analysis(analysis_id: int = None, last_30_days: bool = False, limit: int = 10, offset: int = 0):
    conn = get_db_connection()
    cursor = conn.cursor()
    if analysis_id:
        cursor.execute('SELECT * FROM analysis_data WHERE id = ?', (analysis_id,))
        result = cursor.fetchone()
    else:
        query = 'SELECT * FROM analysis_data'
        params = []
        if last_30_days:
            thirty_days_ago = datetime.now() - timedelta(days=30)
            query += ' WHERE created_at >= ?'
            params.append(thirty_days_ago)
        query += ' LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        cursor.execute(query, params)
        result = cursor.fetchall()
    conn.close()
    return result

def update_analysis(analysis_id: int, text: str = None, analysis_result: Dict[str, Any] = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    updates = []
    params = []
    if text:
        updates.append("text = ?")
        params.append(text)
    if analysis_result:
        updates.append("analysis_result = ?")
        params.append(json.dumps(analysis_result))
    if not updates:
        conn.close()
        return False
    params.append(analysis_id)
    cursor.execute(f'''
        UPDATE analysis_data SET {', '.join(updates)} WHERE id = ?
    ''', params)
    conn.commit()
    conn.close()
    return cursor.rowcount > 0

def delete_analysis(analysis_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM analysis_data WHERE id = ?', (analysis_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0

@app.get("/analyses")
async def get_all_analyses(last_30_days: bool = False, limit: int = 10, offset: int = 0):
    try:
        results = read_analysis(last_30_days=last_30_days, limit=limit, offset=offset)
        formatted_results = []
        for row in results:
            if row:
                try:
                    analysis_result = json.loads(row["analysis_result"]) if row["analysis_result"] else {}
                    formatted_results.append({"id": row["id"], "text": row["text"], "analysis_result": analysis_result})
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in analysis_result for ID {row['id']}: {e}")
                    formatted_results.append(dict(row))
        logger.info(f"Fetched {len(formatted_results)} analyses{' (last 30 days)' if last_30_days else ''}")
        return formatted_results
    except Exception as e:
        logger.error(f"Error fetching analyses: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analyses")

@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: int):
    result = read_analysis(analysis_id)
    if result:
        try:
            analysis_result = json.loads(result["analysis_result"]) if result["analysis_result"] else {}
            return {"id": result["id"], "text": result["text"], "analysis_result": analysis_result}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in analysis_result for ID {result['id']}: {e}")
            return dict(result)
    logger.error(f"Analysis not found for ID: {analysis_id}")
    raise HTTPException(status_code=404, detail="Analysis not found")

@app.get("/analyses")
async def get_all_analyses(last_30_days: bool = False):
    try:
        results = read_analysis(last_30_days=last_30_days)
        formatted_results = []
        for row in results:
            if row:
                try:
                    analysis_result = json.loads(row["analysis_result"]) if row["analysis_result"] else {}
                    formatted_results.append({"id": row["id"], "text": row["text"], "analysis_result": analysis_result})
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in analysis_result for ID {row['id']}: {e}")
                    formatted_results.append(dict(row))
        logger.info(f"Fetched {len(formatted_results)} analyses{' (last 30 days)' if last_30_days else ''}")
        return formatted_results
    except Exception as e:
        logger.error(f"Error fetching analyses: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analyses")

@app.put("/analysis/{analysis_id}")
async def update_analysis_endpoint(analysis_id: int, request: TextRequest = None, analysis_result: Dict[str, Any] = None):
    if not request and not analysis_result:
        raise HTTPException(status_code=400, detail="No update data provided")
    text = request.text if request else None
    success = update_analysis(analysis_id, text, analysis_result)
    if success:
        return {"message": "Analysis updated successfully"}
    logger.error(f"Analysis not found for update, ID: {analysis_id}")
    raise HTTPException(status_code=404, detail="Analysis not found")

@app.delete("/analysis/{analysis_id}")
async def delete_analysis_endpoint(analysis_id: int):
    success = delete_analysis(analysis_id)
    if success:
        return {"message": "Analysis deleted successfully"}
    logger.error(f"Analysis not found for deletion, ID: {analysis_id}")
    raise HTTPException(status_code=404, detail="Analysis not found")

@app.post("/analyze")
async def analyze_text_endpoint(request: TextRequest):
    try:
        result = analyzer.analyze_text(request.text)
        analysis_id = create_analysis(request.text, result)
        return {"id": analysis_id, "text": request.text, "analysis_result": result}
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze text")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)