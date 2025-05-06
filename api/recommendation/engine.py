"""Main recommendation engine implementation."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import asyncio
from typing import List, Optional, Dict, Any
import logging

from .models import get_model_handler
from .utils import create_dummy_mappings

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Engine for generating product recommendations."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.models = {
            "sasrec": None,
            "ssept": None,
            "lightgcn": None
        }
        # Maps to store ID mappings for each model
        self.item_maps = {"sasrec": None, "ssept": None, "lightgcn": None}
        self.reverse_item_maps = {"sasrec": None, "ssept": None, "lightgcn": None}
        self.user_maps = {"ssept": None, "lightgcn": None}  # For user IDs
        self.is_loaded = {"sasrec": False, "ssept": False, "lightgcn": False}
        
    async def load_model(self, model_path: str, models_to_load: List[str] = ["sasrec"]) -> None:
        """Load the specified recommendation models and mappings."""
        for model_type in models_to_load:
            try:
                # Get the appropriate model handler
                model_handler = get_model_handler(model_type)
                
                # Load model asynchronously
                model_data = await asyncio.to_thread(
                    model_handler.load_model_files, model_path, model_type
                )
                self.models[model_type] = model_data["model"]
                
                # Store mappings
                if "item_map" in model_data:
                    self.item_maps[model_type] = model_data["item_map"]
                    self.reverse_item_maps[model_type] = model_data["reverse_item_map"]
                
                if "user_map" in model_data and model_type in ["ssept", "lightgcn"]:
                    self.user_maps[model_type] = model_data["user_map"]
                    
                self.is_loaded[model_type] = True
                logger.info(f"{model_type.upper()} model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load {model_type} model: {str(e)}")
                # Fall back to dummy if needed
                if model_type == "sasrec":  # Always need at least SASRec
                    self.models[model_type] = "dummy_model"
                    dummy_mappings = create_dummy_mappings()
                    self.item_maps[model_type] = dummy_mappings["item_map"]
                    self.reverse_item_maps[model_type] = dummy_mappings["reverse_item_map"]
                    self.is_loaded[model_type] = True
                    logger.warning(f"Using dummy model for {model_type}")
                    
    async def get_all_recommendations(
        self, 
        basket_items: List[str],
        user_id: Optional[str] = None,
        num_recommendations: int = 5,
        include_categories: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate and combine recommendations from applicable models."""
        results = {}
        
        # Always get SASRec recommendations (basket-only)
        if self.is_loaded["sasrec"]:
            try:
                from .models.sasrec import get_recommendations
                results["sasrec"] = await get_recommendations(
                    self.models["sasrec"],
                    self.item_maps["sasrec"],
                    self.reverse_item_maps["sasrec"],
                    basket_items,
                    num_recommendations,
                    include_categories
                )
            except Exception as e:
                logger.error(f"Error getting SASRec recommendations: {str(e)}")
                results["sasrec"] = []
        
        # If user_id is provided, get SSEPT and LightGCN recommendations
        if user_id:
            # Get SSEPT recommendations (user + basket)
            if self.is_loaded["ssept"]:
                try:
                    from .models.ssept import get_recommendations
                    results["ssept"] = await get_recommendations(
                        self.models["ssept"],
                        self.item_maps["ssept"],
                        self.reverse_item_maps["ssept"],
                        self.user_maps["ssept"],
                        user_id,
                        basket_items,
                        num_recommendations,
                        include_categories
                    )
                except Exception as e:
                    logger.error(f"Error getting SSEPT recommendations: {str(e)}")
                    results["ssept"] = []
            
            # Get LightGCN recommendations (user-only)
            if self.is_loaded["lightgcn"]:
                try:
                    from .models.lightgcn import get_recommendations
                    results["lightgcn"] = await get_recommendations(
                        self.models["lightgcn"],
                        self.reverse_item_maps["lightgcn"],
                        self.user_maps["lightgcn"],
                        user_id,
                        num_recommendations,
                        include_categories
                    )
                except Exception as e:
                    logger.error(f"Error getting LightGCN recommendations: {str(e)}")
                    results["lightgcn"] = []
        
        return results