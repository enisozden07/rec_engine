import pytest
import requests
import json
import argparse

# Mark this test to be skipped by default
@pytest.mark.skip(reason="This test requires a running API server")
def test_api(host="http://localhost:8000", use_user=False):
    """
    Manually test the recommendation API.

    Args:
        host: The API host URL
        use_user: Whether to include a user_id in the request
    """
    # Test health check endpoint
    print("Testing health check endpoint...")
    health_response = requests.get(f"{host}/")
    print(f"Status code: {health_response.status_code}")
    print(f"Response: {health_response.json()}\n")
    
    # Test recommendations endpoint
    print("Testing recommendations endpoint...")
    
    request_data = {
        "basket_items": ["SKU123", "SKU456"],
        "num_recommendations": 5
    }
    
    if use_user:
        request_data["user_id"] = "USER001"
        request_data["category_filter"] = ["Electronics", "Home"]
    
    print(f"Request data: {request_data}")
    
    recommendations_response = requests.post(
        f"{host}/recommendations", 
        json=request_data
    )
    
    print(f"Status code: {recommendations_response.status_code}")
    print("Response:")
    print(json.dumps(recommendations_response.json(), indent=2))

# Add a way to run this manually when needed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the recommendation API.")
    parser.add_argument("--host", default="http://localhost:8000", help="API host URL")
    parser.add_argument("--user", action="store_true", help="Include user_id in request")
    
    args = parser.parse_args()
    test_api(args.host, args.user)
