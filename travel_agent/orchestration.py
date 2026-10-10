"""Agent orchestration for parsing, retrieval, tools, and grounded recommendations."""

from __future__ import annotations

from datetime import date
from typing import Any

from .parser import parse_travel_request
from .retrieval import DestinationRetriever, load_destinations
from .tools import default_tools
from .tracking import RunTracker
from .gemini import GeminiToolOrchestrator


class TravelPlanningAgent:
    def __init__(self, documents: list[dict[str, Any]] | None = None, tools: dict[str, Any] | None = None,
                 tracker: RunTracker | None = None):
        self.retriever = DestinationRetriever(documents or load_destinations())
        self.tools = tools or default_tools()
        self.tracker = tracker or RunTracker()
        self.gemini = GeminiToolOrchestrator(self.tools, self.tracker)

    def plan(self, request: str | dict[str, Any]) -> dict[str, Any]:
        parsed = parse_travel_request(request) if isinstance(request, str) else request
        destination = parsed.get("destination") or "a destination matching your interests"
        query = " ".join(filter(None, [destination, *parsed.get("activities", [])]))
        retrieved = self.retriever.search(query, top_k=3)
        start = date.fromisoformat(parsed["start_date"]) if parsed.get("start_date") else None
        end = date.fromisoformat(parsed["end_date"]) if parsed.get("end_date") else None
        days = (end - start).days + 1 if start and end and end >= start else 3
        with self.tracker.run("travel_plan"):
            self.tracker.log_retrieval(query, retrieved)
            gemini = self.gemini.plan(parsed.get("raw_text", ""), parsed, retrieved)
            if gemini["mode"] == "google_gemini" and gemini.get("tool_results"):
                weather_data = gemini["tool_results"].get("weather", {})
                activities_data = gemini["tool_results"].get("activity_search", {})
                budget_data = gemini["tool_results"].get("budget", {})
                providers = {
                    "weather": gemini.get("tool_providers", {}).get("weather", "gemini_tool"),
                    "activities": gemini.get("tool_providers", {}).get("activity_search", "gemini_tool"),
                    "budget": gemini.get("tool_providers", {}).get("budget", "gemini_tool"),
                }
            else:
                weather = self.tools["weather"].run(destination, parsed.get("start_date"))
                self.tracker.log_tool_execution("weather", {"destination": destination, "start_date": parsed.get("start_date")}, weather.data, weather.provider)
                activities = self.tools["activities"].run(destination, parsed.get("activities", []))
                self.tracker.log_tool_execution("activities", {"destination": destination, "activities": parsed.get("activities", [])}, activities.data, activities.provider)
                budget = self.tools["budget"].run(days, parsed.get("budget"))
                self.tracker.log_tool_execution("budget", {"days": days, "budget": parsed.get("budget")}, budget.data, budget.provider)
                weather_data, activities_data, budget_data = weather.data, activities.data, budget.data
                providers = {"weather": weather.provider, "activities": activities.provider, "budget": budget.provider}
            self.tracker.log({"destination": destination, "intent": parsed.get("intent")}, {"retrieval_top_score": retrieved[0]["retrieval_score"] if retrieved else 0.0})
            self.tracker.log_metadata({"agent_mode": gemini["mode"], "model": gemini.get("model", "offline")})
        recommendation = gemini.get("recommendation") or (
            f"Consider {retrieved[0]['name']} for {', '.join(parsed.get('activities') or ['a balanced trip'])}. "
            "The suggestions below are grounded in the destination knowledge base."
            if retrieved else "No matching destination was found; use the tool results as a starting point."
        )
        return {
            "request": parsed,
            "recommendation": recommendation,
            "provider_mode": gemini["mode"],
            "provider_detail": gemini.get("reason"),
            "retrieval": retrieved,
            "weather": weather_data,
            "activities": activities_data,
            "budget": budget_data,
            "providers": providers,
        }
