"""Agent orchestration for parsing, retrieval, tools, and grounded recommendations."""

from __future__ import annotations

from datetime import date
from typing import Any

from .parser import parse_travel_request
from .retrieval import DestinationRetriever, load_destinations
from .tools import default_tools
from .tracking import RunTracker


class TravelPlanningAgent:
    def __init__(self, documents: list[dict[str, Any]] | None = None, tools: dict[str, Any] | None = None,
                 tracker: RunTracker | None = None):
        self.retriever = DestinationRetriever(documents or load_destinations())
        self.tools = tools or default_tools()
        self.tracker = tracker or RunTracker()

    def plan(self, request: str | dict[str, Any]) -> dict[str, Any]:
        parsed = parse_travel_request(request) if isinstance(request, str) else request
        destination = parsed.get("destination") or "a destination matching your interests"
        query = " ".join(filter(None, [destination, *parsed.get("activities", [])]))
        retrieved = self.retriever.search(query, top_k=3)
        start = date.fromisoformat(parsed["start_date"]) if parsed.get("start_date") else None
        end = date.fromisoformat(parsed["end_date"]) if parsed.get("end_date") else None
        days = (end - start).days + 1 if start and end and end >= start else 3
        with self.tracker.run("travel_plan"):
            weather = self.tools["weather"].run(destination, parsed.get("start_date"))
            activities = self.tools["activities"].run(destination, parsed.get("activities", []))
            budget = self.tools["budget"].run(days, parsed.get("budget"))
            self.tracker.log({"destination": destination, "intent": parsed.get("intent")}, {"retrieval_top_score": retrieved[0]["retrieval_score"] if retrieved else 0.0})
        recommendation = (
            f"Consider {retrieved[0]['name']} for {', '.join(parsed.get('activities') or ['a balanced trip'])}. "
            "The suggestions below are grounded in the destination knowledge base."
            if retrieved else "No matching destination was found; use the tool results as a starting point."
        )
        return {
            "request": parsed,
            "recommendation": recommendation,
            "retrieval": retrieved,
            "weather": weather.data,
            "activities": activities.data,
            "budget": budget.data,
            "providers": {"weather": weather.provider, "activities": activities.provider, "budget": budget.provider},
        }
