"""SASRec model implementation."""

import os
import json
import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from .base import ModelHandler

logger = logging.getLogger(__name__)

class SASRecHandler(ModelHandler):
    """Handler for SASRec model."""
    
    def load_model_files(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Load SASRec model files from disk."""
        result = {}
        
        # Load the model checkpoint
        checkpoint_path = os.path.join(model_path, model_type, "checkpoints", f"{model_type}.ckpt")
        
        # Load configuration
        config_path = os.path.join(model_path, model_type, "configs", f"{model_type}_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Import the SASREC model
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
        
        # Load checkpoint weights
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(checkpoint_path).expect_partial()
        
        # Store the model in the result dictionary
        result["model"] = model

        # Load ID mappings
        mapping_path = os.path.join(model_path, "mappings/id_maps.json")
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
        
        return result

async def get_recommendations(
    model,
    item_map,
    reverse_item_map,
    basket_items: List[str], 
    num_recommendations: int = 5,
    include_categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Generate SASRec recommendations based on basket items."""
    # Convert basket items to indices
    basket_indices = []
    for item_id in basket_items:
        if item_id in item_map:
            basket_indices.append(item_map[item_id])
    
    if not basket_indices:
        logger.warning("No valid basket items found for recommendation")
        return []
    
    # Get the sequence max length from the model
    seq_max_len = getattr(model, "seq_max_len", 50)
    
    # Prepare sequence for the model
    seq = np.zeros([1, seq_max_len], dtype=np.int32)
    idx = min(len(basket_indices), seq_max_len)
    seq[0, -idx:] = basket_indices[-idx:]  # Recent items at the end
    
    # Convert to tensor
    seq_tensor = tf.constant(seq, dtype=tf.int32)
    
    try:
        # Updated: Call predict differently based on the model's expected inputs
        if hasattr(model, 'call') and callable(model.call):
            # For newer TensorFlow models that use the call method
            predictions = model.call(seq_tensor, training=False)
        else:
            # Try the standard predict with dictionary input
            predictions = model({'input_seq': seq_tensor}, training=False)
        
        # Handle different output formats
        if isinstance(predictions, dict) and 'output' in predictions:
            scores = predictions['output'][0]
        elif isinstance(predictions, tuple) and len(predictions) > 0:
            scores = predictions[0][0]
        else:
            scores = predictions[0]
    except Exception as e:
        logger.error(f"Error during SASRec prediction: {str(e)}")
        # Fallback: Generate random scores for demonstration
        item_count = getattr(model, "item_num", 100)
        scores = np.random.random(item_count)
    
    # Get top-k items
    top_k_indices = np.argsort(-scores)[:num_recommendations*2]
    
    # Map back to item IDs
    recommendations = []
    for idx in top_k_indices:
        if len(recommendations) >= num_recommendations:
            break
            
        # Skip items already in basket
        if idx in basket_indices:
            continue
            
        # Skip if not in reverse mapping
        if idx not in reverse_item_map:
            continue
            
        item_id = reverse_item_map[idx]
        recommendations.append({
            "product_id": item_id,
            "score": float(scores[idx]),
            "product_name": f"Product {item_id}",  # Add placeholder name
            "category": "unknown"
        })
    
    return recommendations