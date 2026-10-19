"""Stage 5: create backend/test_payload.json from the first row of data/sample_inputs.csv."""
from pathlib import Path
import csv
import json

import joblib

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")
features = bundle["features"]

with (ROOT / "data/sample_inputs.csv").open(encoding="utf-8", newline="") as f:
    row = next(csv.DictReader(f))

missing = [name for name in features if name not in row]
if missing:
    raise ValueError(f"Sample CSV missing columns: {missing}")

payload = {name: row[name] for name in features}  # expected_* column is NOT sent
out = ROOT / "backend/test_payload.json"
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Created {out} with {len(payload)} features")
