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
        logger.info(f"Loading LightGCN model files from model_path: {model_path}, model_type: {model_type}")
        result = {}
        
        # Load embeddings
        user_embeddings_path = os.path.join(model_path, model_type, "user_embeddings.csv")
        item_embeddings_path = os.path.join(model_path, model_type, "item_embeddings.csv")
        logger.info(f"LightGCN user embeddings path: {user_embeddings_path}")
        logger.info(f"LightGCN item embeddings path: {item_embeddings_path}")
        
        user_embeddings = pd.read_csv(user_embeddings_path)
        item_embeddings = pd.read_csv(item_embeddings_path) 
        logger.info(f"Loaded LightGCN user embeddings shape: {user_embeddings.shape}")
        logger.info(f"Loaded LightGCN item embeddings shape: {item_embeddings.shape}")
        
        # Create model (a simple wrapper for embeddings)
        result["model"] = LightGCNEmbeddings(user_embeddings, item_embeddings)
        
        return result
    
    def predict(self, model, user_idx):
        """Calculate scores for all items for a given user."""
        logger.debug(f"LightGCN predict called for user_idx: {user_idx} (type: {type(user_idx)})")
        try:
            # Ensure user_idx is an integer
            if not isinstance(user_idx, (int, np.integer)):
                logger.debug(f"Converting user_idx {user_idx} to int.")
                user_idx = int(user_idx)
                
            # Get user embedding
            if hasattr(model.user_embeddings, 'loc') and user_idx in model.user_embeddings.index:
                # If the index exists, get it directly
                u_vec = model.user_embeddings.loc[user_idx].values
                logger.debug(f"User embedding for {user_idx} found via .loc. Shape: {u_vec.shape}")
            else:
                # Try to find the user in the data frame
                user_id_col = model.user_embeddings.index.name or model.user_embeddings.columns[0]
                logger.debug(f"User embedding for {user_idx} not in index, searching by column: {user_id_col}")
                user_data = model.user_embeddings[model.user_embeddings[user_id_col] == user_idx]
                
                if user_data.empty:
                    logger.warning(f"User index {user_idx} not found in embeddings. Returning random scores.")
                    # Return random scores for all items as fallback
                    return np.random.rand(len(model.item_embeddings))
                    
                # Get embedding columns (exclude ID column)
                embedding_cols = [col for col in user_data.columns if col != user_id_col]
                u_vec = user_data[embedding_cols].values.flatten()
                logger.debug(f"User embedding for {user_idx} found via column search. Shape: {u_vec.shape}")
            
            # Ensure it's a 1D array
            if len(u_vec.shape) > 1:
                logger.debug(f"Flattening u_vec from shape {u_vec.shape}")
                u_vec = u_vec.flatten()
                
            # Get item embeddings and compute dot product
            item_embeddings_values = model.item_embeddings.values
            logger.debug(f"Original item_embeddings shape: {item_embeddings_values.shape}, u_vec shape: {u_vec.shape}")
            
            # Check if the first column of item_embeddings might be an ID
            if item_embeddings_values.shape[1] == u_vec.shape[0] + 1:
                 logger.debug("Item embeddings might have an ID column. Slicing item_embeddings[:, 1:].")
                 item_embeddings_for_dot = item_embeddings_values[:, 1:]
            elif item_embeddings_values.shape[1] != u_vec.shape[0]:
                logger.warning(
                    f"Dimension mismatch for dot product: item_embeddings ({item_embeddings_values.shape[1]}) "
                    f"vs u_vec ({u_vec.shape[0]}). Attempting to use all item_embeddings columns."
                )
                item_embeddings_for_dot = item_embeddings_values # proceed with caution or raise error
            else:
                item_embeddings_for_dot = item_embeddings_values

            if item_embeddings_for_dot.shape[1] != u_vec.shape[0]:
                logger.error(
                    f"Corrected item_embeddings dimension ({item_embeddings_for_dot.shape[1]}) "
                    f"still does not match u_vec dimension ({u_vec.shape[0]}). Returning random scores."
                )
                return np.random.rand(len(model.item_embeddings))

            # Calculate scores using dot product
            logger.debug(f"Performing dot product with item_embeddings shape: {item_embeddings_for_dot.shape} and u_vec shape: {u_vec.shape}")
            scores = np.dot(item_embeddings_for_dot, u_vec)
            logger.debug(f"Calculated scores shape: {scores.shape}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Prediction error for user_idx {user_idx}: {str(e)}", exc_info=True)
            return np.random.rand(len(model.item_embeddings))

async def get_recommendations(
    model,
    user_idx: int,
    num_recommendations: int = 5,
    include_categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Generate LightGCN recommendations based on user index."""
    logger.info(
        f"LightGCN get_recommendations called with user_idx: {user_idx}, "
        f"num_recommendations: {num_recommendations}"
    )
    # Get predictions
    scores = LightGCNHandler().predict(model, user_idx)
    logger.debug(f"LightGCN scores shape for user {user_idx}: {scores.shape if hasattr(scores, 'shape') else 'N/A'}")
    
    # Get top items
    # Ensure scores is 1D array for argsort
    if scores.ndim > 1:
        logger.warning(f"LightGCN scores array is not 1D (shape: {scores.shape}), attempting to flatten.")
        scores = scores.flatten()

    top_indices = np.argsort(-scores)[:num_recommendations*2]
    logger.debug(f"LightGCN top_indices (before filtering): {top_indices}")
    
    # Build recommendations with indices only
    recommendations = []
    for idx in top_indices:
        if len(recommendations) >= num_recommendations:
            break
        
        recommendations.append({
            "index": int(idx),
            "score": float(scores[idx])
        })
    logger.info(f"LightGCN generated {len(recommendations)} recommendations: {recommendations}")
    return recommendations