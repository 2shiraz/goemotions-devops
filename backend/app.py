"""Stage 6: Flask REST API for the GoEmotions sentiment model.

GET  /health   -> service status and model metadata
POST /predict  -> {"text": "..."}  ->  {"prediction": "Positive", "scores": {...}}
"""
from flask import Flask, jsonify, request

from inference import MAX_TEXT_LENGTH, load_bundle, predict_texts

BUNDLE = load_bundle()
FEATURES = BUNDLE["features"]           # ["text"]
TASK = BUNDLE.get("task", "classification")
CLASSES = [str(c) for c in BUNDLE["label_encoder"].classes_]

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK,
                   model=BUNDLE.get("model_name"), classes=CLASSES,
                   model_version=BUNDLE.get("model_version"))


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400

    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch",
                       missing=missing, extra=extra), 400

    text = data["text"]
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Invalid input",
                       detail="'text' must be a non-empty string"), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify(error="Invalid input",
                       detail=f"'text' must be at most {MAX_TEXT_LENGTH} characters"), 400

    try:
        result = predict_texts(BUNDLE, [text])[0]
        result.update({
            "task": TASK,
            "model": BUNDLE.get("model_name"),
            "disclaimer": "Research/teaching prototype only. Scores are "
                          "LinearSVC decision values, not probabilities.",
        })
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
