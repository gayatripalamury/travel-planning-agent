from travel_agent.orchestration import TravelPlanningAgent
from travel_agent.tools import BudgetTool


def test_budget_is_deterministic_and_respects_limit():
    result = BudgetTool().run(3, 1000).data
    assert result["total"] == 579
    assert result["within_budget"] is True


def test_agent_returns_grounded_plan_and_tool_outputs():
    result = TravelPlanningAgent().plan("Plan a food trip to Lisbon from 2026-05-01 to 2026-05-04")
    assert result["retrieval"][0]["name"] == "Lisbon"
    assert result["providers"]["weather"] == "mock"
    assert "budget" in result


def test_agent_counts_start_and_end_dates_inclusive():
    result = TravelPlanningAgent().plan(
        "Plan a food and culture trip to Kyoto from 2026-10-10 to 2026-10-14"
    )
    assert result["budget"]["days"] == 5
