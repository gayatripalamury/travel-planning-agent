from app import create_app


def test_plan_endpoint():
    client = create_app().test_client()
    response = client.post("/api/plan", json={"text": "Plan a culture trip to Kyoto"})
    assert response.status_code == 200
    assert response.json["retrieval"][0]["name"] == "Kyoto"
