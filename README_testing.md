# Testing the Loyalty Recommendation API

This document describes how to test the Loyalty Recommendation API.

## Running Automated Tests

We use pytest for automated testing. To run the tests:

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run from the project root directory
pytest tests/
```

## Manual Testing

For manual testing, use the provided test script:

```bash
# Start the API server in one terminal
uvicorn api.app:app --reload

# In another terminal, run the test script
python tests/manual_test.py

# To include a user ID in the request
python tests/manual_test.py --user

# To test against a different host
python tests/manual_test.py --host http://api.example.com
```

## API Endpoints

1. Health Check: `GET /`
2. Recommendations: `POST /recommendations`

Example POST request body for recommendations:
```json
{
  "basket_items": ["SKU123", "SKU456"],
  "user_id": "USER001",  // Optional
  "num_recommendations": 5,
  "category_filter": ["Electronics"]  // Optional
}
```
