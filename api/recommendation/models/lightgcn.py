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
        
        # Load embeddings
        user_embeddings = pd.read_csv(os.path.join(model_path, model_type, "user_embeddings.csv"))
        item_embeddings = pd.read_csv(os.path.join(model_path, model_type, "item_embeddings.csv")) 
        
        # Create model (a simple wrapper for embeddings)
        result["model"] = LightGCNEmbeddings(user_embeddings, item_embeddings)
        
        return result
    
    def predict(self, model, user_idx):
        """Calculate scores for all items for a given user."""
        try:
            # Ensure user_idx is an integer
            if not isinstance(user_idx, (int, np.integer)):
                user_idx = int(user_idx)
                
            # Get user embedding
            if hasattr(model.user_embeddings, 'loc') and user_idx in model.user_embeddings.index:
                # If the index exists, get it directly
                u_vec = model.user_embeddings.loc[user_idx].values
            else:
                # Try to find the user in the data frame
                user_id_col = model.user_embeddings.index.name or model.user_embeddings.columns[0]
                user_data = model.user_embeddings[model.user_embeddings[user_id_col] == user_idx]
                
                if user_data.empty:
                    logger.warning(f"User index {user_idx} not found in embeddings")
                    # Return random scores for all items as fallback
                    return np.random.rand(len(model.item_embeddings))
                    
                # Get embedding columns (exclude ID column)
                embedding_cols = [col for col in user_data.columns if col != user_id_col]
                u_vec = user_data[embedding_cols].values.flatten()
            
            # Ensure it's a 1D array
            if len(u_vec.shape) > 1:
                u_vec = u_vec.flatten()
                
            # Get item embeddings and compute dot product
            item_embeddings = model.item_embeddings.values
            if len(item_embeddings.shape) > 1 and item_embeddings.shape[1] != u_vec.shape[0]:
                # If dimensions don't match, exclude the ID column or adjust dimensions
                if item_embeddings.shape[1] > u_vec.shape[0]:
                    item_embeddings = item_embeddings[:, 1:] if item_embeddings.shape[1] - 1 == u_vec.shape[0] else item_embeddings
                
            # Calculate scores using dot product
            scores = np.dot(item_embeddings, u_vec)
            
            return scores
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return np.random.rand(len(model.item_embeddings))

async def get_recommendations(
    model,
    user_idx: int,
    num_recommendations: int = 5,
    include_categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Generate LightGCN recommendations based on user index."""
    # Get predictions
    scores = LightGCNHandler().predict(model, user_idx)
    
    # Get top items
    top_indices = np.argsort(-scores)[:num_recommendations*2]
    
    # Build recommendations with indices only
    recommendations = []
    for idx in top_indices:
        if len(recommendations) >= num_recommendations:
            break
        
        recommendations.append({
            "index": int(idx),
            "score": float(scores[idx])
        })
    
    return recommendations