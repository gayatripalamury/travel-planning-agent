# Integration guide

## Runtime modes

The application always supports an offline mode. `TravelPlanningAgent` retrieves destination context with the local TF-IDF vector store and uses deterministic mock tools unless optional providers are configured.

When both `GEMINI_API_KEY` and the `google-genai` package are available, `GeminiToolOrchestrator` sends the parsed request and retrieved context to Gemini and executes Gemini's `weather`, `activity_search`, and `budget` function calls locally. The response includes `provider_mode: google_gemini`. SDK/key errors fall back to `provider_mode: offline`.

## Tool contracts

`WeatherTool.run(destination, start_date)`, `ActivitySearchTool.run(destination, activities)`, and `BudgetTool.run(days, budget, travelers)` return `ToolResult` objects. Inject custom providers in Python or set the documented HTTP provider URLs.

## Observability

`RunTracker` is a no-op when MLflow is unavailable. With `MLFLOW_ENABLED=true`, it records the run, parsed metadata, retrieval query and ranked documents, tool arguments/results/providers, and agent mode/model. Use a local file tracking URI for development.

## Claims model

The claims module trains a deterministic `LogisticRegression` model at import time from the bundled synthetic dataset. It combines model probability and rule guardrails, and returns positive feature contributions as explanations. Outputs are decision support only and require qualified review.
