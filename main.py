# ============================================
# main.py — API FastAPI pour Sentiment Analysis
# ============================================

# Imports des bibliothèques nécessaires (entraînement / divers)
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve)
import re
import string
import warnings
from typing import List, Dict, Optional
import time
import joblib
import os
from pydantic import BaseModel, Field
from datetime import datetime
import nltk
nltk.download('wordnet')    # Pour la lemmatisation (forme de base)

# --- Imports standard / typing
import json
import hashlib

# --- FastAPI & schémas
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware  # <-- CORS

# --- Sklearn / Pipeline
from sklearn.pipeline import make_pipeline

# (Optionnel) LIME pour l'explicabilité
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

# =========================================================
# Config : chemins des artefacts (adapter si nécessaire)
# =========================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "sentiment_model.joblib")
VECT_PATH  = os.environ.get("VECT_PATH", "tfidf_vectorizer.joblib")

# =========================================================
# Schémas Pydantic (définis AVANT les endpoints)
# =========================================================
class TweetRequest(BaseModel):
    text: str = Field(..., max_length=280, description="Texte (ex: tweet) à analyser")

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probability_positive: float
    probability_negative: float

class ExplanationItem(BaseModel):
    word: str
    weight: float

class ExplanationResponse(BaseModel):
    sentiment: str
    explanation: List[ExplanationItem]          # Mots + importances
    html_explanation: Optional[str] = ""        # HTML LIME (si demandé côté front)

# =========================================================
# App FastAPI + CORS
# =========================================================
app = FastAPI(title="💬 API Sentiment Analysis", version="1.0")

# --- CORS ---
# En prod, remplace "*" par les domaines autorisés (séparés par des virgules)
# ex: ALLOWED_ORIGINS="https://ton-app.streamlit.app,https://ton-domaine.com"
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
origins = ["*"] if ALLOWED_ORIGINS.strip() == "*" else [
    o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, OPTIONS, ...
    allow_headers=["*"],   # Authorization, Content-Type, ...
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>💬 API Sentiment Analysis</title>
            <style>
                body {
                    font-family: Arial;
                    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                    color: white;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 30px;
                    border-radius: 12px;
                }
                h1 { text-align: center; }
                .box {
                    background: rgba(255, 255, 255, 0.15);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }
                a {
                    color: white;
                    display: block;
                    padding: 10px;
                    background: rgba(255,255,255,0.2);
                    margin: 5px 0;
                    text-decoration: none;
                    border-radius: 5px;
                    text-align: center;
                }
                a:hover { background: rgba(255,255,255,0.3); }
            </style>
        </head>
        <body>
            <h1>💬 API Sentiment Analysis</h1>
            <div class="box">
                <h3>🧠 Endpoints disponibles :</h3>
                <a href="/docs">📖 Swagger UI (documentation interactive)</a>
                <a href="/health">🩺 Vérification de santé</a>
            </div>
            <div class="box">
                <p>POST /predict — Prédire le sentiment</p>
                <p>POST /explain — Explication LIME de la prédiction</p>
            </div>
        </body>
    </html>
    """

# =========================================================
# Utilitaires
# =========================================================
def preprocess_text(text: str) -> str:
    """
    Prétraitement minimal (adaptable à ton pipeline d'entraînement).
    Si ton modèle joblib est un Pipeline incluant TF-IDF, tu peux
    ne presque rien faire ici (juste strip).
    """
    t = text.strip().lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)   # URLs
    t = re.sub(r"@\w+", " ", t)               # mentions
    t = re.sub(r"#", " ", t)                  # hashtags
    t = re.sub(r"[^a-zà-ÿ0-9\s']", " ", t)    # ponctuation forte
    t = re.sub(r"\s+", " ", t).strip()
    return t

def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()

# =========================================================
# Chargement des artefacts (au démarrage)
# =========================================================
model = None
vectorizer = None
lime_explainer = None

def _load_artifacts() -> None:
    global model, vectorizer, lime_explainer

    # Si le modèle est un Pipeline (TF-IDF inclus), VECT_PATH peut être inutilisé.
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Erreur chargement modèle: {e}")
        model = None

    try:
        if os.path.exists(VECT_PATH):
            vectorizer = joblib.load(VECT_PATH)
        else:
            vectorizer = None
    except Exception as e:
        print(f"⚠️ Erreur chargement vectorizer: {e}")
        vectorizer = None

    if LIME_AVAILABLE:
        try:
            # Classe 0 = négatif, 1 = positif (adapter à tes classes si besoin)
            lime_explainer = LimeTextExplainer(class_names=["Négatif", "Positif"])
        except Exception as e:
            print(f"⚠️ LIME indisponible: {e}")
            lime_explainer = None

_load_artifacts()

# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    """
    Vérifie l'état de santé de l'API et le chargement des artefacts.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_path": MODEL_PATH,
        "vectorizer_path": VECT_PATH if vectorizer is not None else None,
        "lime_available": bool(lime_explainer),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TweetRequest):
    """
    Prédit le sentiment d'un texte.
    - Si `model` est un Pipeline (incluant TF-IDF), `vectorizer` peut être None.
    - Sinon, on applique `vectorizer.transform`.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    text = request.text
    clean = preprocess_text(text)

    # Cas 1 : pipeline (TF-IDF dans le modèle)
    try:
        if hasattr(model, "predict_proba"):
            # Si vectorizer est None, on tente direct (pipeline)
            if vectorizer is None:
                proba = model.predict_proba([clean])[0]
            else:
                X = vectorizer.transform([clean])
                proba = model.predict_proba(X)[0]
        else:
            # Pas de predict_proba → on infère la classe puis on met confidence=1.0
            if vectorizer is None:
                pred = model.predict([clean])[0]
            else:
                pred = model.predict(vectorizer.transform([clean]))[0]
            # On simule des probas “dures”
            if str(pred) in ("1", "Positif", "positive", "Positive"):
                proba = np.array([0.0, 1.0])
            else:
                proba = np.array([1.0, 0.0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {e}")

    sentiment = "Positif" if float(proba[1]) >= 0.5 else "Négatif"
    return PredictionResponse(
        sentiment=sentiment,
        confidence=float(np.max(proba)),
        probability_positive=float(proba[1]),
        probability_negative=float(proba[0]),
    )

@app.post("/explain", response_model=ExplanationResponse)
def explain(request: TweetRequest):
    """
    Explique la prédiction d'un texte avec LIME (si disponible).
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    if not lime_explainer:
        raise HTTPException(status_code=503, detail="LIME non disponible sur ce serveur")

    clean = preprocess_text(request.text)

    # 🚨 Garde-fou : si le texte est vide après nettoyage (ex: emojis only)
    if not clean.strip():
        return ExplanationResponse(
            sentiment="Inconnu",
            explanation=[],
            html_explanation="<div>Aucune explication disponible (texte vide ou non analysable).</div>"
        )

    # Fonction proba compatible LIME
    def predict_proba_fn(texts: List[str]):
        processed = [preprocess_text(t) for t in texts]
        if hasattr(model, "predict_proba"):
            if vectorizer is None:
                return model.predict_proba(processed)
            X = vectorizer.transform(processed)
            return model.predict_proba(X)
        # Si pas de predict_proba, LIME text attend des probabilités → on “binarise”
        if vectorizer is None:
            preds = model.predict(processed)
        else:
            preds = model.predict(vectorizer.transform(processed))
        probs = []
        for p in preds:
            if str(p) in ("1", "Positif", "positive", "Positive"):
                probs.append([0.0, 1.0])
            else:
                probs.append([1.0, 0.0])
        return np.array(probs)

    # Explication
    exp = lime_explainer.explain_instance(
        clean,
        predict_proba_fn,
        num_features=10
    )

    # Classe prédite
    if hasattr(model, "predict_proba"):
        if vectorizer is None:
            proba = model.predict_proba([clean])[0]
        else:
            proba = model.predict_proba(vectorizer.transform([clean]))[0]
        pred_label = 1 if float(proba[1]) >= 0.5 else 0
    else:
        if vectorizer is None:
            pred = model.predict([clean])[0]
        else:
            pred = model.predict(vectorizer.transform([clean]))[0]
        pred_label = 1 if str(pred) in ("1", "Positif", "positive", "Positive") else 0

    sentiment = "Positif" if pred_label == 1 else "Négatif"

    return ExplanationResponse(
        sentiment=sentiment,
        explanation=[ExplanationItem(word=w, weight=float(v)) for w, v in exp.as_list()],
        html_explanation=exp.as_html(),
    )
