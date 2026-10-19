# GoEmotions Sentiment Prediction App

A full-stack ML teaching project. A Linear SVM trained on the GoEmotions Reddit-comment dataset is served by a Flask REST API. Users interact with it through an Angular form. GitHub Actions tests and builds the project on every push, and GitHub Pages hosts the static frontend.

> **Research/teaching prototype only.** Predictions are not suitable for any real-world decision about a person.

## Problem and target

- **Task:** single-label, 4-class text classification
- **Input:** one raw text field, `text` (a comment, up to 5,000 characters)
- **Output classes:** `Positive`, `Negative`, `Neutral`, `Ambiguous`. These come from the GoEmotions sentiment hierarchy, which groups the 28 emotion labels:
  - Positive: admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
  - Negative: anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
  - Ambiguous: confusion, curiosity, realization, surprise
  - Neutral: neutral
  - Annotations whose top group was tied were removed during training.

## Model provenance

| Item | Value |
|---|---|
| Training notebook | `notebooks/Devops_Lab1.ipynb` (Google Colab) |
| Dataset | GoEmotions, `mrm8488/goemotions` on Hugging Face (211,225 annotation rows, 37 columns) |
| Split | `GroupShuffleSplit` 80/20 grouped by comment `id` (no ID overlap, so no leakage) |
| Features | Cleaning `light_v1` (HTML unescape, URL→`URL`, @user/u/user→`USER`, whitespace), then TF-IDF (1–2 grams, `min_df=3`, `max_df=0.98`, 60,000 features, sublinear tf) |
| Model | `LinearSVC(C=0.1, class_weight="balanced")`, chosen over Logistic Regression by macro F1 |
| Test metrics (Linear SVM) | Accuracy 0.57 · Macro F1 0.54 · Weighted F1 0.58 |
| Saved with | scikit-learn **1.6.1** |
| Original artifact | `backend/models/goemotions_best_model.pkl` |
| Deployable bundle | `backend/models/model_bundle.joblib`, built by `backend/package_existing.py` |

The bundle holds the **already-fitted** vectorizer, model and label encoder. Nothing is re-fitted at prediction time. `backend/inference.py` repeats the exact training-time `clean_text()`.

LinearSVC has no `predict_proba`, so the API returns per-class **decision scores** (higher means more confident). These are not probabilities.

## Project structure

```
.
├── backend/
│   ├── app.py                 Flask API (GET /health, POST /predict)
│   ├── inference.py           cleaning + saved vectorizer/model adapter
│   ├── package_existing.py    builds model_bundle.joblib from the Colab .pkl
│   ├── make_payload.py        builds test_payload.json from data/sample_inputs.csv
│   ├── test_pipeline.py       direct inference check vs notebook outputs
│   ├── test_api.py            calls the running server with requests
│   ├── test_payload.json
│   ├── requirements.txt
│   ├── models/                goemotions_best_model.pkl, model_bundle.joblib
│   └── tests/                 test_api.py (6 tests), test_structure.py (2 tests)
├── data/sample_inputs.csv     2 non-sensitive examples with notebook outputs
├── docs/                      environment.txt, evidence screenshots
├── frontend/                  Angular 22 app (form, service, proxy.conf.json)
├── notebooks/Devops_Lab1.ipynb
└── .github/workflows/         ci.yml, pages.yml
```

## Versions

- Python 3.12, with packages pinned in `backend/requirements.txt` (scikit-learn 1.6.1 must match the saved model)
- Node.js 24 LTS (Angular 22 needs Node ≥ 22.22.3 or ≥ 24.15)
- Angular CLI 22.2.x

## Install (Windows PowerShell, from the project root)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

npm install -g @angular/cli
cd frontend
npm install
cd ..
```

If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.

## Run locally (keep both terminals open)

**Terminal 1, Flask** (from the project root):
```powershell
.\.venv\Scripts\Activate.ps1
python backend\app.py
```
Check that http://127.0.0.1:5000/health returns `"status": "ok"`.

**Terminal 2, Angular** (from `frontend`):
```powershell
cd frontend
ng serve --open --proxy-config proxy.conf.json
```
Open http://localhost:4200, type a comment, and click **Run local prediction**.

**Terminal 3** is for tests and Git.

## Tests

```powershell
python backend\test_pipeline.py                 # direct inference, PASS expected
python -m pytest -q backend\tests               # 8 passed
python backend\test_api.py                      # needs Flask running
curl.exe --noproxy "*" -i http://localhost:4200/api/health   # proxy check
cd frontend; npx ng test --watch=false; npm run build
```

## API

`POST /predict` with body `{"text": "This is disappointing and makes me angry."}` returns:

```json
{
  "prediction": "Negative",
  "scores": {"Ambiguous": -1.248, "Negative": 1.262, "Neutral": -0.917, "Positive": -1.169},
  "task": "classification",
  "model": "Linear SVM",
  "disclaimer": "Research/teaching prototype only. ..."
}
```

Any of these return HTTP 400: a missing `text` field, extra fields, empty text, text over 5,000 characters, or invalid JSON.

## CI/CD

- **`ci.yml`** runs on every push or PR to `main`:
  - backend job: structure tests, direct inference check and Flask API tests (the 2.5 MB model is committed, so CI runs the model tests)
  - frontend job: `npm ci`, unit tests, production build, and the `angular-build` artifact
- **`pages.yml`** builds Angular with `--base-href /<repo-name>/` (read automatically) and deploys to GitHub Pages.

## Links

- GitHub repository: `https://github.com/<USERNAME>/<REPO>`
- GitHub Pages: `https://<USERNAME>.github.io/<REPO>/`

## Deployment limitations

GitHub Pages hosts **only the static Angular files**. It cannot run Python, Flask or the model. On Pages, the form still renders, but the predict button is disabled and a notice explains why. Live predictions work only when Flask and Angular run locally. Public live predictions would need a separately hosted HTTPS backend.

## Ethics and limitations

- GoEmotions is English Reddit text. The model may not transfer well to other domains, dialects or languages.
- The labels are crowd-sourced and subjective, and test accuracy is about 57%, so many predictions will be wrong.
- Do not use this model to judge, profile or make decisions about real people.
