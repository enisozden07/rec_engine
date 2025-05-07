from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ProductRecommendation(BaseModel):
    """Model for a product recommendation."""
    product_id: str
    score: float
    product_name: Optional[str] = None  # Optional product name
    category: Optional[str] = None  # Optional category field

class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    basket_items: List[str]  # Required
    user_id: Optional[str] = None  # Optional user ID
    num_recommendations: int = 5
    category_filter: Optional[List[str]] = None

class ModelRecommendations(BaseModel):
    """Recommendations from a specific model."""
    model_name: str
    recommendations: List[ProductRecommendation]

class RecommendationResponse(BaseModel):
    """Response model with recommendations from multiple models."""
    basket_items: List[str]
    user_id: Optional[str] = None
    model_recommendations: List[ModelRecommendations]