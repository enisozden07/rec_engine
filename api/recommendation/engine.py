"""Main recommendation engine implementation."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import asyncio
from typing import List, Optional, Dict, Any
import logging

from .models import get_model_handler
from .utils import load_id_mappings, map_user_id, map_item_ids, map_recommendations_to_original_ids

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
        # Maps to store ID mappings
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.is_loaded = {"sasrec": False, "ssept": False, "lightgcn": False}

    async def load_mappings(self, model_path: str) -> None:
        """Load mappings from CSV files."""
        mappings_dir = os.path.join(model_path, "mappings")
        
        try:
            self.item_id_to_idx, self.idx_to_item_id, self.user_id_to_idx, self.idx_to_user_id = \
                load_id_mappings(mappings_dir)
            
            # Log the size of the mappings for debugging
            logger.info(f"Loaded {len(self.item_id_to_idx)} item mappings and {len(self.user_id_to_idx)} user mappings")
            logger.info(f"First few item mappings: {list(self.item_id_to_idx.items())[:5]}")
            logger.info(f"First few user mappings: {list(self.user_id_to_idx.items())[:5]}")
            
            # Verify reverse mappings match
            if len(self.item_id_to_idx) != len(self.idx_to_item_id):
                logger.warning(f"Item mapping size mismatch: {len(self.item_id_to_idx)} != {len(self.idx_to_item_id)}")
                
            if len(self.user_id_to_idx) != len(self.idx_to_user_id):
                logger.warning(f"User mapping size mismatch: {len(self.user_id_to_idx)} != {len(self.idx_to_user_id)}")
        
        except Exception as e:
            logger.error(f"Error loading mappings: {str(e)}")
    
    async def load_model(self, model_path: str, models_to_load: List[str] = ["sasrec"]) -> None:
        """Load the specified recommendation models and mappings."""
        # Load mappings first - shared across all models
        await self.load_mappings(model_path)

        for model_type in models_to_load:
            try:
                # Get the appropriate model handler
                model_handler = get_model_handler(model_type)
                
                # Load model asynchronously
                model_data = await asyncio.to_thread(
                    model_handler.load_model_files, model_path, model_type
                )
                self.models[model_type] = model_data["model"]
                self.is_loaded[model_type] = True
                logger.info(f"{model_type.upper()} model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load {model_type} model: {str(e)}")
                    
    async def get_all_recommendations(
        self, 
        basket_items: List[str],
        user_id: Optional[str] = None,
        num_recommendations: int = 5,
        include_categories: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate and combine recommendations from applicable models."""
        # Map IDs at the entry point
        mapped_user_id = map_user_id(user_id, self.user_id_to_idx) if user_id else None
        mapped_basket_ids = map_item_ids(basket_items, self.item_id_to_idx)
        
        results = {}
        
        # Get SASRec recommendations (basket-only)
        if self.is_loaded["sasrec"] and mapped_basket_ids:
            try:
                from .models.sasrec import get_recommendations
                raw_recs = await get_recommendations(
                    self.models["sasrec"],
                    mapped_basket_ids,
                    num_recommendations,
                    include_categories
                )
                
                # Map indices back to original IDs
                results["sasrec"] = map_recommendations_to_original_ids(raw_recs, self.idx_to_item_id)
            except Exception as e:
                logger.error(f"Error getting SASRec recommendations: {str(e)}")
                results["sasrec"] = []
        
        # If user_id is provided, get SSEPT and LightGCN recommendations
        if mapped_user_id is not None:
            # Get SSEPT recommendations (user + basket)
            if self.is_loaded["ssept"] and mapped_basket_ids:
                try:
                    from .models.ssept import get_recommendations
                    raw_recs = await get_recommendations(
                        self.models["ssept"],
                        mapped_user_id,
                        mapped_basket_ids,
                        num_recommendations,
                        include_categories
                    )
                    
                    # Map indices back to original IDs
                    results["ssept"] = map_recommendations_to_original_ids(raw_recs, self.idx_to_item_id)
                except Exception as e:
                    logger.error(f"Error getting SSEPT recommendations: {str(e)}")
                    results["ssept"] = []
            
            # Get LightGCN recommendations (user-only)
            if self.is_loaded["lightgcn"]:
                try:
                    from .models.lightgcn import get_recommendations
                    raw_recs = await get_recommendations(
                        self.models["lightgcn"],
                        mapped_user_id,
                        num_recommendations,
                        include_categories
                    )
                    
                    # Map indices back to original IDs
                    results["lightgcn"] = map_recommendations_to_original_ids(raw_recs, self.idx_to_item_id)
                except Exception as e:
                    logger.error(f"Error getting LightGCN recommendations: {str(e)}")
                    results["lightgcn"] = []
        
        return results