"""Optional Google Gemini function-calling orchestration.

The Google Gen AI SDK is imported lazily so the default application remains
offline-first and does not require credentials or network access.
"""

from __future__ import annotations

import json
import os
from typing import Any

TOOL_SCHEMAS = [
    {
        "name": "weather",
        "description": "Get a forecast for a destination and optional start date.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "start_date": {"type": "string"},
            },
            "required": ["destination"],
        },
    },
    {
        "name": "activity_search",
        "description": "Find activities matching interests in a destination.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "activities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["destination"],
        },
    },
    {
        "name": "budget",
        "description": "Estimate trip costs for days, travelers, and optional budget.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer"},
                "budget": {"type": "number"},
                "travelers": {"type": "integer"},
            },
            "required": ["days"],
        },
    },
]


class GeminiToolOrchestrator:
    def __init__(self, tools: dict[str, Any], tracker: Any = None, client: Any = None,
                 api_key: str | None = None, model: str | None = None):
        self.tools = tools
        self.tracker = tracker
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.client = client
        self.unavailable_reason: str | None = None
        if client is None:
            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                self.unavailable_reason = "GEMINI_API_KEY is not configured"
            else:
                try:
                    from google import genai
                    self.client = genai.Client(api_key=key)
                except ImportError:
                    self.unavailable_reason = "google-genai package is not installed"
        self.enabled = self.client is not None

    @property
    def mode(self) -> str:
        return "google_gemini" if self.enabled else "offline"

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def _execute(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name == "weather":
            result = self.tools["weather"].run(arguments["destination"], arguments.get("start_date"))
        elif name == "activity_search":
            result = self.tools["activities"].run(arguments["destination"], arguments.get("activities", []))
        elif name == "budget":
            result = self.tools["budget"].run(
                int(arguments.get("days", 1)), arguments.get("budget"),
                int(arguments.get("travelers", 1)),
            )
        else:
            raise ValueError(f"Unknown Gemini tool: {name}")
        if self.tracker:
            self.tracker.log_tool_execution(name, arguments, result.data, result.provider)
        return result.data, result.provider

    @staticmethod
    def _parts(response: Any) -> list[Any]:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            return list(getattr(content, "parts", None) or [])
        return []

    def plan(self, request_text: str, parsed: dict[str, Any],
             retrieved: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.enabled:
            return {"mode": self.mode, "reason": self.unavailable_reason}
        prompt = (
            f"Travel request: {request_text}\nParsed request: {json.dumps(parsed)}\n"
            f"Retrieved destination context (ground recommendations in this context): "
            f"{json.dumps(retrieved, ensure_ascii=False)}\n"
            "Create a practical plan. Call weather, activity_search, and budget tools "
            "before writing the final recommendation. Do not invent tool results."
        )
        tool_results: dict[str, dict[str, Any]] = {}
        providers: dict[str, str] = {}
        contents: list[Any] = [prompt]
        try:
            from google.genai import types
            tool = types.Tool(function_declarations=TOOL_SCHEMAS)
            config = types.GenerateContentConfig(
                system_instruction="You are a travel planning agent grounded in retrieved context.",
                tools=[tool],
            )
            for _ in range(5):
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                calls = [part.function_call for part in self._parts(response)
                         if getattr(part, "function_call", None)]
                if not calls:
                    text = getattr(response, "text", "") or ""
                    return {"mode": self.mode, "model": self.model,
                            "recommendation": text.strip(), "tool_results": tool_results,
                            "tool_providers": providers}
                contents.append(getattr(response, "candidates")[0].content)
                function_parts = []
                for call in calls:
                    data, provider = self._execute(call.name, dict(call.args or {}))
                    tool_results[call.name] = data
                    providers[call.name] = provider
                    function_parts.append(types.Part.from_function_response(
                        name=call.name, response={"result": data}
                    ))
                contents.append(types.Content(role="user", parts=function_parts))
        except Exception as exc:
            return {"mode": "offline", "reason": f"Gemini request failed: {exc}"}
        return {"mode": "offline", "reason": "Gemini did not produce a final response"}
