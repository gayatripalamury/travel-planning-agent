from travel_agent.parser import parse_travel_request


def test_parser_extracts_core_fields():
    result = parse_travel_request("Plan a food and hiking trip to Banff from 2026-07-10 to 2026-07-15 under $2,000")
    assert result["intent"] == "plan_trip"
    assert result["destination"] == "Banff"
    assert result["activities"] == ["hiking", "food"]
    assert result["budget"] == 2000
    assert result["start_date"] == "2026-07-10"
