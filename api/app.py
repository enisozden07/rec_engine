from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging

from .models import RecommendationRequest, RecommendationResponse, ModelRecommendations
from .recommendation import RecommendationEngine
from .config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# Initialize app
app = FastAPI(
    title="Loyalty Recommendation API",
    description="API for product recommendations based on customer purchase history",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize recommendation engine as a singleton
engine = RecommendationEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize recommendation engine on startup."""
    try:
        # Determine which models to load
        models_to_load = ["sasrec"]  # Always load SASRec
        
        # If user-based models are enabled in config
        if settings.enable_user_models:
            models_to_load.extend(["ssept", "lightgcn"])
            
        await engine.load_model(
            model_path=settings.model_path,
            models_to_load=models_to_load
        )
        
        logger.info(f"Recommendation engine initialized with models: {', '.join(models_to_load)}")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {str(e)}")

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": engine.is_loaded}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Generate recommendations using multiple models based on inputs."""
    try:
        model_input_log = f"basket items: {len(request.basket_items)}"
        if request.user_id:
            model_input_log += f", user_id: {request.user_id}"
            
        logger.info(f"Processing recommendation request with {model_input_log}")
        
        # Get recommendations from all applicable models
        all_recommendations = await engine.get_all_recommendations(
            basket_items=request.basket_items,
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            include_categories=request.category_filter
        )
        
        # Format into response structure
        model_recommendations = []
        for model_name, recommendations in all_recommendations.items():
            if recommendations:  # Only include models that returned recommendations
                model_recommendations.append(
                    ModelRecommendations(
                        model_name=model_name, 
                        recommendations=recommendations
                    )
                )
        
        return RecommendationResponse(
            basket_items=request.basket_items,
            user_id=request.user_id,
            model_recommendations=model_recommendations
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )