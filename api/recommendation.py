import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import asyncio
import json
from typing import List, Optional, Dict, Any
import logging
import tensorflow as tf  # Add TensorFlow import instead of PyTorch
import numpy as np
import pandas as pd

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
                # Load model asynchronously
                model_data = await asyncio.to_thread(
                    self._load_model_files, model_path, model_type
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
                    self.item_maps[model_type] = {"P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
                    self.reverse_item_maps[model_type] = {1: "P1", 2: "P2", 3: "P3", 4: "P4", 5: "P5"}
                    self.is_loaded[model_type] = True
                    logger.warning(f"Using dummy model for {model_type}")
            
    def _load_model_files(self, model_path: str, model_type: str) -> Any:
        """Load model files from disk - this runs in a separate thread."""
        result = {}
        
        # Special handling for LightGCN which only has embeddings
        if model_type == "lightgcn":
            # Load user embeddings
            user_embed_path = os.path.join(model_path, model_type, "user_embeddings.csv")
            user_embeddings = pd.read_csv(user_embed_path)
            
            # Load item embeddings
            item_embed_path = os.path.join(model_path, model_type, "item_embeddings.csv")
            item_embeddings = pd.read_csv(item_embed_path)
            
            # Create user and item maps
            user_map = {str(user_id): idx for idx, user_id in enumerate(user_embeddings['user_id'].values)}
            item_map = {str(item_id): idx for idx, item_id in enumerate(item_embeddings['item_id'].values)}
            
            # Create model (a simple wrapper for embeddings)
            class LightGCNEmbeddings:
                def __init__(self, user_embeddings, item_embeddings):
                    self.user_embeddings = user_embeddings
                    self.item_embeddings = item_embeddings
            
            # Return all components
            result["model"] = LightGCNEmbeddings(user_embeddings, item_embeddings)
            result["item_map"] = item_map
            result["reverse_item_map"] = {v: k for k, v in item_map.items()}
            result["user_map"] = user_map
            
            return result
        
        # Load the model checkpoint
        checkpoint_path = os.path.join(model_path, model_type, "checkpoints", f"{model_type}.ckpt")
        
        # Load configuration
        config_path = os.path.join(model_path, model_type, "configs", f"{model_type}_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create and load the appropriate model based on model_type
        if model_type == "sasrec":
            # Import the SASREC model - update the import path as needed
            from recommenders.models.sasrec.model import SASREC
            
            model = SASREC(
                item_num=config["item_num"],
                seq_max_len=config["seq_max_len"],
                num_blocks=config["num_blocks"],
                embedding_dim=config["embedding_dim"],
                attention_dim=config["attention_dim"],
                attention_num_heads=config["attention_num_heads"],
                dropout_rate=config["dropout_rate"],
                conv_dims=config["conv_dims"],
                l2_reg=config["l2_reg"],
                num_neg_test=config["num_neg_test"]
            )
        elif model_type == "ssept":
            # Import the SSEPT model - update the import path as needed
            from recommenders.models.sasrec.ssept import SSEPT
            
            model = SSEPT(
                item_num=config["item_num"],
                user_num=config.get("user_num", 1000),  # Default if not in config
                seq_max_len=config["seq_max_len"],
                num_blocks=config["num_blocks"],
                embedding_dim=config["embedding_dim"],
                user_embedding_dim=config.get("user_embedding_dim", 10),
                item_embedding_dim=config.get("item_embedding_dim", config["embedding_dim"]),
                attention_dim=config["attention_dim"],
                attention_num_heads=config["attention_num_heads"],
                dropout_rate=config["dropout_rate"],
                conv_dims=config["conv_dims"],
                l2_reg=config["l2_reg"],
                num_neg_test=config["num_neg_test"]
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint weights
        # Note: The actual loading code depends on how your models are defined
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(checkpoint_path).expect_partial()
        
        # Store the model in the result dictionary
        result["model"] = model

        # Load ID mappings
        mapping_path = os.path.join(model_path, model_type, "mappings/id_maps.json")
        with open(mapping_path, "r") as f:
            mappings = json.load(f)
            # Adjust keys based on actual mapping structure
            item_map = None
            if "item_id_map" in mappings:
                item_map = mappings["item_id_map"]
            elif "item_id_to_idx" in mappings:
                item_map = mappings["item_id_to_idx"]
            else:
                # Try to find appropriate mapping key
                for key in mappings:
                    if "item" in key.lower() and "id" in key.lower():
                        item_map = mappings[key]
                        break
            
            # Create reverse mapping
            reverse_item_map = {v: k for k, v in item_map.items()}
            
            # Add to result dictionary
            result["item_map"] = item_map
            result["reverse_item_map"] = reverse_item_map
            
            # For user models, also get user map
            if model_type in ["ssept", "lightgcn"] and "user_id_map" in mappings:
                result["user_map"] = mappings["user_id_map"]
        
        return result
        
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
                results["sasrec"] = await self.get_sasrec_recommendations(
                    basket_items, num_recommendations, include_categories
                )
            except Exception as e:
                logger.error(f"Error getting SASRec recommendations: {str(e)}")
                results["sasrec"] = []
        
        # If user_id is provided, get SSEPT and LightGCN recommendations
        if user_id:
            # Get SSEPT recommendations (user + basket)
            if self.is_loaded["ssept"]:
                try:
                    results["ssept"] = await self.get_ssept_recommendations(
                        user_id, basket_items, num_recommendations, include_categories
                    )
                except Exception as e:
                    logger.error(f"Error getting SSEPT recommendations: {str(e)}")
                    results["ssept"] = []
            
            # Get LightGCN recommendations (user-only)
            if self.is_loaded["lightgcn"]:
                try:
                    results["lightgcn"] = await self.get_lightgcn_recommendations(
                        user_id, num_recommendations, include_categories
                    )
                except Exception as e:
                    logger.error(f"Error getting LightGCN recommendations: {str(e)}")
                    results["lightgcn"] = []
        
        return results
    
    async def get_sasrec_recommendations(
        self, 
        basket_items: List[str], 
        num_recommendations: int = 5,
        include_categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate SASRec recommendations based on basket items only."""
        if not self.is_loaded["sasrec"]:
            raise RuntimeError("SASRec model not loaded")
        
        # Convert basket items to indices
        basket_indices = []
        for item_id in basket_items:
            if item_id in self.item_maps["sasrec"]:
                basket_indices.append(self.item_maps["sasrec"][item_id])
        
        if not basket_indices:
            logger.warning("No valid basket items found for recommendation")
            return []
        
        # Get the sequence max length from the model
        seq_max_len = getattr(self.models["sasrec"], "seq_max_len", 50)
        
        # Prepare sequence for the model - put items at the end of sequence
        seq = np.zeros([1, seq_max_len], dtype=np.int32)
        idx = min(len(basket_indices), seq_max_len)
        seq[0, -idx:] = basket_indices[-idx:]  # Recent items at the end
        
        # Convert to tensor
        seq_tensor = tf.constant(seq, dtype=tf.int32)
        
        # Get predictions using SASRec model
        predictions = self.models["sasrec"].predict(seq_tensor)
        scores = predictions[0]
        
        # Get top-k items
        top_k_indices = np.argsort(-scores)[:num_recommendations*2]  # Get 2x to filter later
        
        # Map back to item IDs
        recommendations = []
        for idx in top_k_indices:
            if len(recommendations) >= num_recommendations:
                break
                
            # Skip items already in basket
            if idx in basket_indices:
                continue
                
            # Skip if not in reverse mapping
            if idx not in self.reverse_item_maps["sasrec"]:
                continue
                
            item_id = self.reverse_item_maps["sasrec"][idx]
            recommendations.append({
                "product_id": item_id,
                "score": float(scores[idx]),
                "category": "unknown"  # Add category if available
            })
        
        return recommendations

    async def get_ssept_recommendations(
        self, 
        user_id: str,
        basket_items: List[str], 
        num_recommendations: int = 5,
        include_categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate SSEPT recommendations based on both user ID and basket."""
        if not self.is_loaded["ssept"]:
            raise RuntimeError("SSEPT model not loaded")
        
        # Convert user_id to model index
        if user_id not in self.user_maps["ssept"]:
            logger.warning(f"User {user_id} not found in SSEPT model")
            return []
            
        user_idx = self.user_maps["ssept"][user_id]
        
        # Convert basket items to indices
        basket_indices = []
        for item_id in basket_items:
            if item_id in self.item_maps["ssept"]:
                basket_indices.append(self.item_maps["ssept"][item_id])
        
        # Prepare sequence for the model
        seq = np.zeros([1, self.models["ssept"].seq_max_len], dtype=np.int32)
        idx = min(len(basket_indices), self.models["ssept"].seq_max_len)
        seq[0, -idx:] = basket_indices[-idx:]
        
        # Convert to tensor and add user information
        seq_tensor = tf.constant(seq, dtype=tf.int32)
        user_tensor = tf.constant([[user_idx]], dtype=tf.int32)
        
        # Get predictions using SSEPT model
        # ...implementation for SSEPT predictions...

    async def get_lightgcn_recommendations(
        self, 
        user_id: str,
        num_recommendations: int = 5,
        include_categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate LightGCN recommendations based on user ID only."""
        if not self.is_loaded["lightgcn"]:
            raise RuntimeError("LightGCN model not loaded")
        
        # Convert user_id to model index
        if user_id not in self.user_maps["lightgcn"]:
            logger.warning(f"User {user_id} not found in LightGCN model")
            return []
        
        user_idx = self.user_maps["lightgcn"][user_id]
        model = self.models["lightgcn"]
        
        # Get user embedding
        user_embedding = model.user_embeddings.iloc[user_idx, 1:].values  # Skip user_id column
        
        # Compute dot product with all item embeddings
        item_embeddings = model.item_embeddings.iloc[:, 1:].values  # Skip item_id column
        scores = np.dot(user_embedding, item_embeddings.T)
        
        # Get top-k items
        top_k_indices = np.argsort(-scores)[:num_recommendations]
        
        # Map back to item IDs
        recommendations = []
        for idx in top_k_indices:
            item_id = self.reverse_item_maps["lightgcn"][idx]
            recommendations.append({
                "product_id": item_id,
                "score": float(scores[idx]),
                "category": "unknown"  # You may want to add category info if available
            })
        
        return recommendations