"""Tool interfaces with deterministic mock providers and optional HTTP integrations."""

from __future__ import annotations

import os
import json
from urllib.parse import urlencode
from urllib.request import urlopen
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    tool: str
    data: dict[str, Any]
    provider: str = "mock"


class WeatherTool:
    name = "weather"

    def __init__(self, provider: Callable[[str, str | None], dict[str, Any]] | None = None):
        self.provider = provider

    def run(self, destination: str, start_date: str | None = None) -> ToolResult:
        if self.provider:
            return ToolResult(self.name, self.provider(destination, start_date), "custom")
        return ToolResult(self.name, {
            "destination": destination,
            "forecast": "Mild conditions expected; pack layers and check local alerts.",
            "high_celsius": 24,
            "precipitation_risk": "moderate",
            "date": start_date,
        })


class ActivitySearchTool:
    name = "activities"

    def __init__(self, provider: Callable[[str, list[str]], list[dict[str, Any]]] | None = None):
        self.provider = provider

    def run(self, destination: str, activities: list[str]) -> ToolResult:
        if self.provider:
            return ToolResult(self.name, {"items": self.provider(destination, activities)}, "custom")
        requested = activities or ["culture", "food", "nature"]
        items = [
            {"name": f"{activity.title()} experience in {destination}", "category": activity, "estimated_cost": 35}
            for activity in requested
        ]
        return ToolResult(self.name, {"items": items}, "mock")


class BudgetTool:
    name = "budget"

    def run(self, days: int, budget: float | None, travelers: int = 1) -> ToolResult:
        days = max(days, 1)
        estimate = {
            "lodging": round(95 * days * travelers, 2),
            "food": round(45 * days * travelers, 2),
            "activities": round(35 * days * travelers, 2),
            "local_transport": round(18 * days * travelers, 2),
        }
        total = round(sum(estimate.values()), 2)
        return ToolResult(self.name, {
            "estimate": estimate,
            "total": total,
            "requested_budget": budget,
            "within_budget": budget is None or total <= budget,
            "days": days,
            "travelers": travelers,
        })


def default_tools() -> dict[str, Any]:
    weather_url = os.getenv("WEATHER_API_URL")
    activities_url = os.getenv("ACTIVITIES_API_URL")

    def weather_provider(destination: str, start_date: str | None) -> dict[str, Any]:
        query = urlencode({"destination": destination, "date": start_date or ""})
        with urlopen(f"{weather_url}?{query}", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def activities_provider(destination: str, activities: list[str]) -> list[dict[str, Any]]:
        query = urlencode({"destination": destination, "activities": ",".join(activities)})
        with urlopen(f"{activities_url}?{query}", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("items", payload) if isinstance(payload, dict) else payload

    return {
        "weather": WeatherTool(weather_provider if weather_url else None),
        "activities": ActivitySearchTool(activities_provider if activities_url else None),
        "budget": BudgetTool(),
    }
