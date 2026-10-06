from travel_agent.claims import triage_claim


def test_claims_triage_explains_high_risk():
    result = triage_claim({
        "amount": 12000, "prior_claims_30d": 4, "provider_risk": "high",
        "duplicate_invoice": True,
    })
    assert result["triage"] == "high"
    assert result["reasons"]
    assert "not a fraud determination" in result["disclaimer"]
