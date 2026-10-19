"""Stage 4: package the trusted Colab artifact into backend/models/model_bundle.joblib.

Source: goemotions_best_model.pkl saved by notebooks/Devops_Lab1.ipynb (cell 41)
with scikit-learn 1.6.1. The fitted TF-IDF vectorizer, LinearSVC and
LabelEncoder are kept exactly as trained; only metadata is added.
"""
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "models" / "goemotions_best_model.pkl"
TARGET = ROOT / "models" / "model_bundle.joblib"

source = joblib.load(SOURCE)  # trusted artifact only
for key in ("model", "vectorizer", "label_encoder"):
    assert key in source, f"Missing '{key}' in source artifact"
assert hasattr(source["model"], "predict"), "Model must implement predict"
assert hasattr(source["vectorizer"], "vocabulary_"), "Vectorizer must be fitted"

bundle = dict(source)
bundle.update({
    "features": ["text"],
    "task": "classification",
    "model_version": "1.0",
    "sklearn_version": "1.6.1",
})
joblib.dump(bundle, TARGET, compress=3)
print("Saved model bundle:", TARGET.name, "| model:", bundle["model_name"],
      "| classes:", list(bundle["label_encoder"].classes_))
