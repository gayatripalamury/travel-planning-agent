"""Flask entry point for the travel planning agent."""

from __future__ import annotations

import os
from flask import Flask, jsonify, render_template, request

from travel_agent.claims import triage_claim
from travel_agent.orchestration import TravelPlanningAgent
from travel_agent.parser import parse_travel_request
from travel_agent.retrieval import DestinationRetriever, load_destinations
from travel_agent.tracking import RunTracker


def create_app() -> Flask:
    app = Flask(__name__)
    tracker = RunTracker(enabled=os.getenv("MLFLOW_ENABLED", "").lower() in {"1", "true", "yes"})
    agent = TravelPlanningAgent(tracker=tracker)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/parse")
    def parse():
        payload = request.get_json(silent=True) or {}
        return jsonify(parse_travel_request(payload.get("text", "")))

    @app.post("/api/plan")
    def plan():
        payload = request.get_json(silent=True) or {}
        if not payload.get("text") and not payload.get("request"):
            return jsonify({"error": "Provide text or request."}), 400
        return jsonify(agent.plan(payload.get("text") or payload["request"]))

    @app.post("/api/retrieve")
    def retrieve():
        payload = request.get_json(silent=True) or {}
        query = payload.get("query", "")
        results = DestinationRetriever(load_destinations()).search(query, payload.get("top_k", 3))
        return jsonify({"results": results})

    @app.post("/api/claims/triage")
    def claims():
        return jsonify(triage_claim(request.get_json(silent=True) or {}))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"})
