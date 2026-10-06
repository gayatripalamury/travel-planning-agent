from travel_agent.retrieval import DestinationRetriever, load_destinations


def test_retrieval_prefers_matching_destination():
    results = DestinationRetriever(load_destinations()).search("mountain hiking wildlife")
    assert results[0]["name"] == "Banff"
    assert results[0]["retrieval_score"] > 0
