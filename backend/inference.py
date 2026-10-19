"""Model-specific inference adapter for the GoEmotions sentiment model.

The saved bundle is not a single sklearn Pipeline: it holds a fitted
TfidfVectorizer, a fitted LinearSVC and a fitted LabelEncoder. This module
reproduces the exact text cleaning used in training (notebooks/Devops_Lab1.ipynb,
cleaning_version "light_v1") and chains the saved, already-fitted objects.
Nothing is re-fitted at prediction time.
"""
from pathlib import Path
import html
import re

import joblib

BASE = Path(__file__).resolve().parent
BUNDLE_PATH = BASE / "models" / "model_bundle.joblib"
MAX_TEXT_LENGTH = 5000


def clean_text(text):
    """Identical to clean_text() in the training notebook."""
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"(?:u/|@)\w+", " USER ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_bundle(path=BUNDLE_PATH):
    bundle = joblib.load(path)  # trusted artifact created by the project owner
    if bundle.get("cleaning_version") != "light_v1":
        raise ValueError("Unexpected cleaning_version; check the model bundle")
    return bundle


def predict_texts(bundle, texts):
    """Return one result dict per input text."""
    cleaned = [clean_text(t) for t in texts]
    features = bundle["vectorizer"].transform(cleaned)
    model = bundle["model"]
    encoded = model.predict(features)
    labels = bundle["label_encoder"].inverse_transform(encoded)
    classes = [str(c) for c in bundle["label_encoder"].classes_]
    scores = model.decision_function(features)
    results = []
    for label, row in zip(labels, scores):
        results.append({
            "prediction": str(label),
            # LinearSVC has no predict_proba; these are signed decision scores
            # (higher = more confident), NOT probabilities.
            "scores": {c: round(float(s), 4) for c, s in zip(classes, row)},
        })
    return results
