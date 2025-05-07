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
        
        # Determine ID columns
        user_id_col = 'user_id' if 'user_id' in user_embeddings.columns else user_embeddings.columns[0]
        item_id_col = 'item_id' if 'item_id' in item_embeddings.columns else item_embeddings.columns[0]
        
        # Set index on embeddings for faster lookup
        if not user_embeddings.index.name:
            user_embeddings.set_index(user_id_col, inplace=True)
        if not item_embeddings.index.name:
            item_embeddings.set_index(item_id_col, inplace=True)
        
        # Create model (a simple wrapper for embeddings)
        result["model"] = LightGCNEmbeddings(user_embeddings, item_embeddings)
        
        return result
    
    def predict(self, model, user_idx):
        """Calculate scores for all items for a given user."""
        try:
            # Get user embedding
            u_vec = (model.user_embeddings.iloc[user_idx].values 
                    if isinstance(user_idx, (int, np.integer))
                    else model.user_embeddings.loc[user_idx].values)
            
            # Ensure it's a 1D array
            if len(u_vec.shape) > 1:
                u_vec = u_vec.flatten()
                
            # Get item embeddings and compute dot product
            item_embeddings = model.item_embeddings.values
            if item_embeddings.shape[1] != u_vec.shape[0]:
                item_embeddings = item_embeddings[:, 1:]
                
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