"""Optional MLflow integration that never prevents an agent run."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


class RunTracker:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.mlflow = None
        if enabled:
            try:
                import mlflow
                self.mlflow = mlflow
            except ImportError:
                self.enabled = False

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
