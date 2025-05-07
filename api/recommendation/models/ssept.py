"""SSEPT model implementation."""

import os
import json
import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from .base import ModelHandler

logger = logging.getLogger(__name__)

class SSEPTHandler(ModelHandler):
    """Handler for SSEPT model."""
    
    def load_model_files(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Load SSEPT model files from disk."""
        result = {}
        
        # Load the model checkpoint
        checkpoint_path = os.path.join(model_path, model_type, "checkpoints", f"{model_type}.ckpt")
        
        # Load configuration
        config_path = os.path.join(model_path, model_type, "configs", f"{model_type}_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Import the SSEPT model
        from recommenders.models.sasrec.ssept import SSEPT
        
        model = SSEPT(
            item_num=config["item_num"],
            user_num=config.get("user_num", 1000),
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
        
        # Load checkpoint weights
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(checkpoint_path).expect_partial()
        
        # Store the model and config
        result["model"] = model
        result["config"] = config
        
        return result
    
    def predict(self, model, inputs):
        """Make predictions with the loaded model."""
        seq_tensor, user_tensor = inputs
        if hasattr(model, 'call') and callable(model.call):
            # Try calling with both tensors as call arguments
            return model.call(seq_tensor, user_tensor, training=False)
        else:
            # Try dictionary-based input
            return model({
                'input_seq': seq_tensor,
                'user_id': user_tensor
            }, training=False)

async def get_recommendations(
    model,
    user_idx: int,
    basket_indices: List[int],
    num_recommendations: int = 5,
    include_categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Generate SSEPT recommendations based on user index and basket item indices."""
    if not basket_indices:
        logger.warning("No valid basket items found for SSEPT recommendation")
        return []
    
    # Get the sequence max length from the model
    seq_max_len = getattr(model, "seq_max_len", 50)
    
    # Prepare sequence for the model
    seq = np.zeros([1, seq_max_len], dtype=np.int32)
    idx = min(len(basket_indices), seq_max_len)
    seq[0, -idx:] = basket_indices[-idx:]
    
    # Convert to tensor and add user information
    seq_tensor = tf.constant(seq, dtype=tf.int32)
    user_tensor = tf.constant([[user_idx]], dtype=tf.int32)
    
    # Get model handler and make prediction
    model_handler = SSEPTHandler()
    predictions = model_handler.predict(model, (seq_tensor, user_tensor))
    
    # Handle different output formats
    if isinstance(predictions, dict) and 'output' in predictions:
        scores = predictions['output'][0]
    elif isinstance(predictions, tuple) and len(predictions) > 0:
        scores = predictions[0][0]
    else:
        scores = predictions[0]
    
    # Get top-k items
    top_k_indices = np.argsort(-scores)[:num_recommendations*2]
    
    # Build recommendations
    recommendations = []
    for idx in top_k_indices:
        if len(recommendations) >= num_recommendations:
            break
            
        # Skip items already in basket
        if idx in basket_indices:
            continue
            
        recommendations.append({
            "index": int(idx),
            "score": float(scores[idx])
        })
    
    return recommendations