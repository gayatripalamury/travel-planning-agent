"""Optional MLflow integration that never prevents an agent run."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


class RunTracker:
    def __init__(self, enabled: bool = False):
        self.requested = enabled
        self.enabled = False
        self.mlflow = None
        if enabled:
            try:
                import mlflow
                self.mlflow = mlflow
                tracking_uri = __import__("os").getenv("MLFLOW_TRACKING_URI")
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                self.enabled = True
            except ImportError:
                pass

    def run(self, name: str):
        if not self.enabled or not self.mlflow:
            return nullcontext()
        return self.mlflow.start_run(run_name=name)

    def log(self, params: dict[str, Any] | None = None, metrics: dict[str, float] | None = None):
        if not self.enabled or not self.mlflow:
            return
        for key, value in (params or {}).items():
            self.mlflow.log_param(key, str(value))
        for key, value in (metrics or {}).items():
            self.mlflow.log_metric(key, value)

    def log_text(self, text: str, artifact_file: str):
        if self.enabled and self.mlflow:
            self.mlflow.log_text(text, artifact_file)

    def log_retrieval(self, query: str, results: list[dict[str, Any]]):
        self.log({"retrieval_query": query, "retrieval_count": len(results)})
        self.log_text(__import__("json").dumps(results, indent=2), "retrieval.json")

    def log_tool_execution(self, name: str, arguments: dict[str, Any], result: dict[str, Any], provider: str):
        self.log({"tool_name": name, "tool_provider": provider})
        self.log_text(__import__("json").dumps({
            "tool": name, "arguments": arguments, "provider": provider, "result": result,
        }, indent=2), f"tools/{name}.json")

    def log_metadata(self, metadata: dict[str, Any]):
        self.log(metadata)
