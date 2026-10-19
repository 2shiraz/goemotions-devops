"""Stage 7: automated positive and negative API tests (need the model bundle)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import app, FEATURES  # noqa: E402

PAYLOAD = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))


def test_health():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["features"] == len(FEATURES)


def test_predict_valid():
    response = app.test_client().post("/predict", json=PAYLOAD)
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["prediction"] == "Positive"
    assert set(body["scores"]) == {"Ambiguous", "Negative", "Neutral", "Positive"}


def test_predict_negative_example():
    response = app.test_client().post(
        "/predict", json={"text": "This is disappointing and makes me angry."})
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "Negative"


def test_predict_missing_feature():
    incomplete = dict(PAYLOAD)
    incomplete.pop(FEATURES[0])
    response = app.test_client().post("/predict", json=incomplete)
    assert response.status_code == 400


def test_predict_empty_text():
    response = app.test_client().post("/predict", json={"text": "   "})
    assert response.status_code == 400


def test_predict_invalid_json():
    response = app.test_client().post("/predict", data="not-json",
                                      content_type="application/json")
    assert response.status_code == 400
