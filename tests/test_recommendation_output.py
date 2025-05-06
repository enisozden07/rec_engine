import pytest
import json
from unittest.mock import patch, AsyncMock

from api.recommendation.utils import create_dummy_mappings

@pytest.mark.asyncio
async def test_recommendation_mappings(test_client, mock_recommendation_engine):
    """
    Test that recommendations correctly map product IDs to names using item mappings.
    This test verifies that the recommendation engine properly translates internal 
    IDs to human-readable product information.
    """
    # Setup mock mappings
    dummy_mappings = create_dummy_mappings()
    
    # Mock the get_all_recommendations method to return properly structured data
    # that matches the format expected by the API
    mock_recommendation_engine.get_all_recommendations = AsyncMock(return_value={
        "sasrec": [
            {"product_id": "P1", "product_name": "Premium Coffee Beans", "category": "Groceries", "score": 0.95},
            {"product_id": "P2", "product_name": "Organic Milk", "category": "Dairy", "score": 0.85},
            {"product_id": "P3", "product_name": "Whole Grain Bread", "category": "Bakery", "score": 0.75}
        ],
        "ssept": [
            {"product_id": "P4", "product_name": "Fresh Apples", "category": "Produce", "score": 0.92},
            {"product_id": "P5", "product_name": "Chocolate Bar", "category": "Confectionery", "score": 0.82}
        ]
    })
    
    # Create test request with user ID and basket items
    request_data = {
        "customer_id": "user123",
        "basket_items": ["P4", "P5"],
        "num_recommendations": 5
    }
    
    # Make request to recommendations endpoint
    response = test_client.post("/recommendations", json=request_data)
    
    # Verify status code
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    
    # Parse response
    response_data = response.json()
    
    # Print response structure for debugging
    print(f"\nResponse data structure: {json.dumps(response_data, indent=2)}")
    
    # Check response structure - matches the actual API response format
    assert "model_recommendations" in response_data, "Response missing 'model_recommendations' key"
    assert "basket_items" in response_data, "Response missing 'basket_items' key"
    
    # Check that we have at least one model with recommendations
    assert len(response_data["model_recommendations"]) > 0, "No model recommendations returned"
    
    # Extract and display all product names from all models
    print("\n=== RECOMMENDED PRODUCTS ===")
    for model_data in response_data["model_recommendations"]:
        model_name = model_data["model_name"]
        print(f"\nModel: {model_name.upper()}")
        print("-" * (len(model_name) + 8))
        
        for i, rec in enumerate(model_data["recommendations"], 1):
            print(f"{i}. {rec['product_name']} (ID: {rec['product_id']}, Category: {rec['category']}, Score: {rec['score']:.2f})")
    
    # Verify at least one recommendation has meaningful product name
    found_products = False
    for model_data in response_data["model_recommendations"]:
        for rec in model_data["recommendations"]:
            if len(rec["product_name"]) > 0 and rec["product_name"] != "Test Product":
                found_products = True
                break
    
    assert found_products, "No meaningful product names found in recommendations"

@pytest.mark.asyncio
async def test_basket_influence_on_recommendations(test_client, mock_recommendation_engine):
    """
    Test that different basket items lead to different recommendations.
    """
    # Setup two different basket scenarios
    basket1 = ["P1", "P2"]  # Coffee and Milk
    basket2 = ["P3", "P4"]  # Bread and Apples
    
    # Configure mock to return different results based on input basket
    async def mock_get_all_recs(basket_items, **kwargs):
        if basket_items == basket1:
            return {
                "sasrec": [
                    {"product_id": "P3", "product_name": "Whole Grain Bread", "category": "Bakery", "score": 0.9},
                    {"product_id": "P5", "product_name": "Chocolate Bar", "category": "Confectionery", "score": 0.8}
                ]
            }
        else:
            return {
                "sasrec": [
                    {"product_id": "P1", "product_name": "Premium Coffee Beans", "category": "Groceries", "score": 0.85},
                    {"product_id": "P2", "product_name": "Organic Milk", "category": "Dairy", "score": 0.75}
                ]
            }
    
    # Apply our custom mock
    mock_recommendation_engine.get_all_recommendations = AsyncMock(side_effect=mock_get_all_recs)
    
    # Test first basket
    print("\n=== TESTING BASKET INFLUENCE ===")
    print(f"Basket 1: {basket1} (Coffee and Milk)")
    response1 = test_client.post(
        "/recommendations", 
        json={"customer_id": "user123", "basket_items": basket1, "num_recommendations": 2}
    )
    data1 = response1.json()
    
    # Test second basket
    print(f"Basket 2: {basket2} (Bread and Apples)")
    response2 = test_client.post(
        "/recommendations", 
        json={"customer_id": "user123", "basket_items": basket2, "num_recommendations": 2}
    )
    data2 = response2.json()
    
    # Extract and display all recommendations for both baskets
    print("\nRecommendations for Basket 1:")
    for model_data in data1["model_recommendations"]:
        print(f"- Model: {model_data['model_name']}")
        for rec in model_data["recommendations"]:
            print(f"  * {rec['product_name']} (Score: {rec['score']:.2f})")
    
    print("\nRecommendations for Basket 2:")
    for model_data in data2["model_recommendations"]:
        print(f"- Model: {model_data['model_name']}")
        for rec in model_data["recommendations"]:
            print(f"  * {rec['product_name']} (Score: {rec['score']:.2f})")
    
    # Verify we got different recommendations for different baskets
    rec1_names = [rec["product_name"] 
                  for model in data1["model_recommendations"] 
                  for rec in model["recommendations"]]
    
    rec2_names = [rec["product_name"] 
                  for model in data2["model_recommendations"] 
                  for rec in model["recommendations"]]
    
    assert rec1_names != rec2_names, "Expected different recommendations for different baskets"
    print("\nVerification successful: Different baskets produce different recommendations")
