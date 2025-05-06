import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app import app
from api.recommendation import RecommendationEngine

@pytest.fixture
def test_client():
    """Return a TestClient for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def mock_recommendation_engine():
    """Create a mock recommendation engine for testing."""
    with patch("api.app.engine") as mock_engine:
        # Configure the mock engine
        mock_engine.is_loaded = True
        mock_engine.get_all_recommendations = AsyncMock(return_value={
            "sasrec": [
                {"product_id": "P001", "product_name": "Test Product 1", "category": "Test Category", "score": 0.95},
                {"product_id": "P002", "product_name": "Test Product 2", "category": "Test Category", "score": 0.85}
            ],
            "ssept": [
                {"product_id": "P003", "product_name": "Test Product 3", "category": "Test Category", "score": 0.75},
                {"product_id": "P004", "product_name": "Test Product 4", "category": "Test Category", "score": 0.65}
            ]
        })
        
        # Return the mock engine
        yield mock_engine
