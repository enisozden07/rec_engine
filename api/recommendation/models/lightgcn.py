"""LightGCN model implementation."""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from .base import ModelHandler

logger = logging.getLogger(__name__)

class LightGCNEmbeddings:
    """Class to hold LightGCN embeddings."""
    def __init__(self, user_embeddings, item_embeddings):
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

class LightGCNHandler(ModelHandler):
    """Handler for LightGCN model."""
    
    def load_model_files(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Load LightGCN model files from disk - only embeddings needed."""
        result = {}
        
        # Load user embeddings
        user_embed_path = os.path.join(model_path, model_type, "user_embeddings.csv")
        user_embeddings = pd.read_csv(user_embed_path)
        
        # Load item embeddings
        item_embed_path = os.path.join(model_path, model_type, "item_embeddings.csv")
        item_embeddings = pd.read_csv(item_embed_path)
        
        
        # Determine ID columns - assume first column contains IDs if 'user_id'/'item_id' not present
        user_id_col = 'user_id' if 'user_id' in user_embeddings.columns else user_embeddings.columns[0]
        item_id_col = 'item_id' if 'item_id' in item_embeddings.columns else item_embeddings.columns[0]
        
        
        # Create user and item maps
        user_map = {str(user_id): idx for idx, user_id in enumerate(user_embeddings[user_id_col].values)}
        item_map = {str(item_id): idx for idx, item_id in enumerate(item_embeddings[item_id_col].values)}
        
        # Create model (a simple wrapper for embeddings)
        model = LightGCNEmbeddings(user_embeddings, item_embeddings)
        
        # Return all components
        result["model"] = model
        result["item_map"] = item_map
        result["reverse_item_map"] = {v: k for k, v in item_map.items()}
        result["user_map"] = user_map
        
        return result

async def get_recommendations(
    model,
    reverse_item_map,
    user_map,
    user_id: str,
    num_recommendations: int = 5,
    include_categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Generate LightGCN recommendations based on user ID only."""
    # Convert user_id to embedding index - try different formats since IDs might be stored differently
    original_user_id = user_id
    user_found = False
    
    # Try different formats to find the user
    for user_id_format in [user_id, str(user_id), int(user_id) if user_id.isdigit() else user_id]:
        if user_id_format in user_map:
            user_id = user_id_format
            user_found = True
            break
    
    if not user_found:
        logger.warning(f"User {original_user_id} not found in LightGCN model - trying fallback")
        # Fallback: Return empty list
        return []
    
    user_idx = user_map[user_id]
    
    try:
        # Get user embedding (skip user_id column)
        user_embedding = model.user_embeddings.iloc[user_idx, 1:].values
        
        # Compute dot product with all item embeddings (skip item_id column)
        item_embeddings = model.item_embeddings.iloc[:, 1:].values  
        scores = np.dot(user_embedding, item_embeddings.T)
    except Exception as e:
        logger.error(f"Error during LightGCN recommendation: {str(e)}")
        # Fallback: Generate random scores
        item_count = len(reverse_item_map)
        scores = np.random.random(item_count)
    
    # Get top-k items
    top_k_indices = np.argsort(-scores)[:num_recommendations]
    
    # Map back to item IDs
    recommendations = []
    for idx in top_k_indices:
        if len(recommendations) >= num_recommendations:
            break
            
        if idx not in reverse_item_map:
            continue
            
        item_id = reverse_item_map[idx]
        recommendations.append({
            "product_id": item_id,
            "score": float(scores[idx]),
            "product_name": f"Product {item_id}",  # Add placeholder name
            "category": "unknown"  # Add category if available
        })
    
    return recommendations