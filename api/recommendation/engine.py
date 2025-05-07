"""Main recommendation engine implementation."""

import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import asyncio
from typing import List, Optional, Dict, Any
import logging

from .models import get_model_handler

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
        try:
            mappings_dir = os.path.join(model_path, "mappings")
            
            # Load user mapping
            user_map_path = os.path.join(mappings_dir, "user_mapping.csv")
            if os.path.exists(user_map_path):
                user_df = pd.read_csv(user_map_path)
                if len(user_df.columns) >= 2:
                    # Assuming first column is original ID and second is index
                    orig_id_col = user_df.columns[0]
                    idx_col = user_df.columns[1]
                    
                    # Create mappings in both directions
                    self.user_id_to_idx = {str(row[orig_id_col]): int(row[idx_col]) 
                                         for _, row in user_df.iterrows()}
                    self.idx_to_user_id = {int(row[idx_col]): str(row[orig_id_col]) 
                                         for _, row in user_df.iterrows()}
                    logger.info(f"Loaded user mappings from {user_map_path}")
            
            # Load item mapping
            item_map_path = os.path.join(mappings_dir, "item_mappings.csv")
            if os.path.exists(item_map_path):
                item_df = pd.read_csv(item_map_path)
                if len(item_df.columns) >= 2:
                    # Assuming first column is original ID and second is index
                    orig_id_col = item_df.columns[0]
                    idx_col = item_df.columns[1]
                    
                    # Create mappings in both directions
                    self.item_id_to_idx = {str(row[orig_id_col]): int(row[idx_col]) 
                                         for _, row in item_df.iterrows()}
                    self.idx_to_item_id = {int(row[idx_col]): str(row[orig_id_col]) 
                                         for _, row in item_df.iterrows()}
                    logger.info(f"Loaded item mappings from {item_map_path}")
                    
        except Exception as e:
            logger.error(f"Failed to load mappings from CSV: {str(e)}")
        
    def map_user_id(self, user_id: str) -> Optional[int]:
        """Map external user ID to internal index."""
        if not user_id:
            return None
            
        # Try different formats
        for id_format in [user_id, str(user_id), int(user_id) if user_id.isdigit() else None]:
            if id_format is not None and str(id_format) in self.user_id_to_idx:
                return self.user_id_to_idx[str(id_format)]
                
        logger.warning(f"User ID {user_id} not found in mappings")
        return None
        
    def map_item_ids(self, item_ids: List[str]) -> List[int]:
        """Map external item IDs to internal indices."""
        mapped_ids = []
        for item_id in item_ids:
            # Try different formats
            for id_format in [item_id, str(item_id), int(item_id) if item_id.isdigit() else None]:
                if id_format is not None and str(id_format) in self.item_id_to_idx:
                    mapped_ids.append(self.item_id_to_idx[str(id_format)])
                    break
        
        return mapped_ids
        
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
        mapped_user_id = self.map_user_id(user_id) if user_id else None
        mapped_basket_ids = self.map_item_ids(basket_items)
        
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
                results["sasrec"] = [
                    {
                        "product_id": self.idx_to_item_id.get(int(rec["index"]), str(rec["index"])),
                        "score": rec["score"],
                        "product_name": f"Product {self.idx_to_item_id.get(int(rec['index']), str(rec['index']))}",
                        "category": rec.get("category", "unknown")
                    }
                    for rec in raw_recs
                ]
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
                    results["ssept"] = [
                        {
                            "product_id": self.idx_to_item_id.get(int(rec["index"]), str(rec["index"])),
                            "score": rec["score"],
                            "product_name": f"Product {self.idx_to_item_id.get(int(rec['index']), str(rec['index']))}",
                            "category": rec.get("category", "unknown")
                        }
                        for rec in raw_recs
                    ]
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
                    results["lightgcn"] = [
                        {
                            "product_id": self.idx_to_item_id.get(int(rec["index"]), str(rec["index"])),
                            "score": rec["score"], 
                            "product_name": f"Product {self.idx_to_item_id.get(int(rec['index']), str(rec['index']))}",
                            "category": rec.get("category", "unknown")
                        }
                        for rec in raw_recs
                    ]
                except Exception as e:
                    logger.error(f"Error getting LightGCN recommendations: {str(e)}")
                    results["lightgcn"] = []
        
        return results