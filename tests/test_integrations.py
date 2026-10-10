from types import SimpleNamespace

from travel_agent.gemini import GeminiToolOrchestrator
from travel_agent.claims import triage_claim
from travel_agent.orchestration import TravelPlanningAgent
from travel_agent.tools import default_tools
from travel_agent.tracking import RunTracker


def test_gemini_falls_back_without_key_or_sdk():
    orchestrator = GeminiToolOrchestrator(default_tools(), api_key=None)
    assert orchestrator.mode == "offline"
    assert orchestrator.plan("trip", {}, [])["mode"] == "offline"


def test_gemini_tool_schema_and_fake_tool_use_loop(monkeypatch):
    class FakeTypes:
        class Tool:
            def __init__(self, **kwargs):
                pass
        class GenerateContentConfig:
            def __init__(self, **kwargs):
                pass
        class Part:
            @staticmethod
            def from_function_response(**kwargs):
                return kwargs
        class Content:
            def __init__(self, **kwargs):
                pass
    monkeypatch.setitem(__import__("sys").modules, "google.genai", SimpleNamespace(types=FakeTypes))

    class FakeClient:
        def __init__(self):
            self.calls = 0

            class Models:
                def __init__(self, outer):
                    self.outer = outer

                def create(self, **kwargs):
                    self.outer.calls += 1
                    if self.outer.calls == 1:
                        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
                            SimpleNamespace(function_call=SimpleNamespace(
                                name="budget", args={"days": 2, "budget": 1000, "travelers": 1}
                            ))
                        ]))])
                    return SimpleNamespace(text="A compact plan.", candidates=[])

                generate_content = create

            self.models = Models(self)

    orchestrator = GeminiToolOrchestrator(default_tools(), client=FakeClient())
    result = orchestrator.plan("two day trip", {"destination": "Kyoto"}, [])
    assert orchestrator.mode == "google_gemini"
    assert result["tool_results"]["budget"]["days"] == 2
    assert "A compact plan" in result["recommendation"]


def test_claim_model_has_explanations():
    result = triage_claim({
        "amount": 12000, "prior_claims_30d": 4, "provider_risk": "high",
        "duplicate_invoice": True,
    })
    assert result["model"] == "sklearn_logistic_regression"
    assert result["model_probability"] > 0.5
    assert result["feature_contributions"]


def test_tracker_disabled_is_safe():
    tracker = RunTracker(enabled=True)
    tracker.log_retrieval("kyoto food", [{"name": "Kyoto", "retrieval_score": 0.9}])
    tracker.log_tool_execution("budget", {"days": 2}, {"total": 386}, "mock")
    tracker.log_metadata({"mode": "offline"})
    assert tracker.enabled is False


def test_agent_explicitly_reports_offline_mode():
    result = TravelPlanningAgent().plan("Plan a food trip to Lisbon")
    assert result["provider_mode"] == "offline"
