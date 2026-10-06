# Offline-first Travel Planning Agent

This project is a small Flask application that turns natural-language travel requests into explainable, retrieval-grounded plans. It runs without API keys or network access by default.

## Features

- Rule-based parsing for intent, destination, dates, activities, budget, and currency.
- Agent orchestration through explicit weather, activity-search, and budget tool interfaces.
- Deterministic mock providers by default; custom providers can be injected in Python without changing orchestration.
- Pure-Python TF-IDF retrieval over `travel_agent/data/destinations.json`, with retrieved facts included in the response for RAG grounding.
- Optional MLflow run, parameter, metric, and retrieval-score tracking. Missing MLflow is handled gracefully.
- Explainable medical-insurance claim triage rules with reasons and a clear decision-support disclaimer.
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

## Optional integrations

Set `MLFLOW_ENABLED=true` and install `mlflow` to enable tracking. External weather or activity services should be connected through provider callables passed to `WeatherTool` or `ActivitySearchTool`; no secrets are committed or required by the default path.

## Tests

```bash
pytest -q
```

The claims module is a triage aid only. It does not determine fraud, deny coverage, or provide medical advice; qualified professionals must review outputs.
