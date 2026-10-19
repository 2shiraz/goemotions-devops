"""Stage 5: direct inference check, compared with the notebook's known outputs."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from inference import load_bundle, predict_texts  # noqa: E402

bundle = load_bundle()
payload = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))
assert set(bundle["features"]) == set(payload), "Payload keys must match features"

result = predict_texts(bundle, [payload["text"]])[0]
print("Prediction:", result["prediction"])
print("Decision scores:", result["scores"])

with (ROOT / "data/sample_inputs.csv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        got = predict_texts(bundle, [row["text"]])[0]["prediction"]
        assert got == row["expected_from_notebook"], (row["text"], got)
        print(f"  matches notebook: {row['text']!r} -> {got}")
print("PASS: local inference")
