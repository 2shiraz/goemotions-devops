"""Stage 13: portable smoke test that does not load the model."""
from pathlib import Path


def test_source_files_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "backend/app.py").exists()
    assert (root / "backend/inference.py").exists()
    assert (root / "frontend/package.json").exists()
    assert (root / "README.md").exists()


def test_clean_text_matches_training():
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "backend"))
    from inference import clean_text
    assert clean_text("Hi u/bob see https://x.com &amp; more") == "Hi USER see URL & more"
