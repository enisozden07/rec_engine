import pytest
from fastapi.testclient import TestClient

def test_health_check(test_client, mock_recommendation_engine):
    """Test the health check endpoint."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_recommendations_endpoint(test_client, mock_recommendation_engine):
    """Test the recommendations endpoint."""
    # Test request with basket items only
    request_data = {
        "basket_items": ["SKU123", "SKU456"],
        "num_recommendations": 5
    }
    
    response = test_client.post("/recommendations", json=request_data)
    assert response.status_code == 200
    
    # Validate response structure
    data = response.json()
    assert "basket_items" in data
    assert "model_recommendations" in data
    assert len(data["model_recommendations"]) > 0
    
    # Test with user_id
    request_data = {
        "basket_items": ["SKU123"],
        "user_id": "USER001",
        "num_recommendations": 5,
        "category_filter": ["Electronics"]
    }
    
    response = test_client.post("/recommendations", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == "USER001"
    assert len(data["model_recommendations"]) > 0
