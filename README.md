# Offline-first Travel Planning Agent

This project is a Flask application that turns natural-language travel requests into explainable, retrieval-grounded plans. It runs without API keys or network access by default, while supporting optional real Google Gemini function-calling, HTTP providers, MLflow tracking, and a reproducible claim-risk model.

## Features

- Rule-based parsing for intent, destination, dates, activities, budget, and currency.
- Agent orchestration through explicit weather, activity-search, and budget tool interfaces.
- Optional Google Gemini function-calling loop. Gemini can call all three tools using JSON schemas; output reports `provider_mode: google_gemini` or `offline`.
- Deterministic mock providers by default; custom providers can be injected in Python without changing orchestration.
- Pure-Python TF-IDF retrieval over `travel_agent/data/destinations.json`, with retrieved facts included in the response for RAG grounding.
- Optional MLflow tracking for runs, parsed request metadata, retrieval queries/results/scores, tool calls, providers, and model metadata. Missing MLflow is handled gracefully.
- Reproducible scikit-learn logistic-regression claim-risk model trained from `travel_agent/data/claims_training.json`, with feature contributions, rule guardrails, and a clear decision-support disclaimer.
- Flask API and simple browser UI.

## Setup and usage

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`. Example API request:

```bash
curl -X POST http://127.0.0.1:5000/api/plan ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Plan a hiking trip to Banff from 2026-07-10 to 2026-07-15 under $1800\"}"
```

Other endpoints are `POST /api/parse`, `POST /api/retrieve`, and `POST /api/claims/triage`.

## Optional integrations and modes

### Google Gemini

Install the optional Google Gen AI SDK and configure the key in your local environment:

```powershell
pip install "google-genai>=1.0,<2"
$env:GEMINI_API_KEY="your-key-in-your-local-environment"
$env:GEMINI_MODEL="gemini-2.0-flash"
python app.py
```

Never commit the key or paste it into chat. If the SDK or key is unavailable, the same request uses deterministic offline orchestration and reports `provider_mode: offline`.

### MLflow

```powershell
pip install "mlflow>=2.0"
$env:MLFLOW_ENABLED="true"
$env:MLFLOW_TRACKING_URI="file:./mlruns"
python app.py
```

Tracking is best-effort and never blocks a plan when MLflow is not installed.

## Tests

```bash
pytest -q
```

The claims module is a triage aid only. It does not determine fraud, deny coverage, or provide medical advice; qualified professionals must review outputs.
