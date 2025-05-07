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
        logger.info(f"Loading SASRec model files from model_path: {model_path}, model_type: {model_type}")
        result = {}
        
        # Load the model checkpoint
        checkpoint_path = os.path.join(model_path, model_type, "checkpoints", f"{model_type}.ckpt")
        logger.info(f"SASRec checkpoint path: {checkpoint_path}")
        
        # Load configuration
        config_path = os.path.join(model_path, model_type, "configs", f"{model_type}_config.json")
        logger.info(f"SASRec config path: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.debug(f"SASRec config loaded: {config}")
        
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
        logger.info(f"Attempting to restore SASRec checkpoint from: {checkpoint_path}")
        status = ckpt.restore(checkpoint_path).expect_partial()
        logger.info(f"SASRec checkpoint restore status: {status}") # Log the status object
        
        # Store the model and config
        result["model"] = model
        result["config"] = config
        
        return result
    
    def predict(self, model, inputs):
        """Make predictions with the loaded model."""
        logger.debug(f"SASRec predict called. Inputs shape: {inputs.shape}")
        if hasattr(model, 'call') and callable(model.call):
            # For newer TensorFlow models that use the call method
            logger.debug("SASRec model has callable 'call' attribute. Using model.call(inputs).")
            return model.call(inputs, training=False)
        else:
            # Try the standard predict with dictionary input
            logger.debug("SASRec model does not have callable 'call' or using dictionary input.")
            return model({'input_seq': inputs}, training=False)

async def get_recommendations(
    model,
    basket_indices: List[int], 
    num_recommendations: int = 5,
    include_categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Generate SASRec recommendations based on basket item indices."""
    logger.info(
        f"SASRec get_recommendations called with basket_indices: {basket_indices}, "
        f"num_recommendations: {num_recommendations}"
    )
    if not basket_indices:
        logger.warning("No valid basket items found for recommendation")
        return []
    
    # Get the sequence max length from the model
    seq_max_len = getattr(model, "seq_max_len", 50)
    logger.debug(f"SASRec model seq_max_len: {seq_max_len}")
    
    # Prepare sequence for the model
    seq = np.zeros([1, seq_max_len], dtype=np.int32)
    idx = min(len(basket_indices), seq_max_len)
    seq[0, -idx:] = basket_indices[-idx:]  # Recent items at the end
    
    # Convert to tensor
    seq_tensor = tf.constant(seq, dtype=tf.int32)
    logger.debug(f"SASRec input seq_tensor: {seq_tensor}")
    
    # Get model handler and make prediction
    model_handler = SASRecHandler()
    predictions = model_handler.predict(model, seq_tensor)
    logger.debug(f"SASRec raw predictions type: {type(predictions)}")
    
    # Handle different output formats
    if isinstance(predictions, dict) and 'output' in predictions:
        scores = predictions['output'][0]
        logger.debug("SASRec predictions extracted from dict['output'][0]")
    elif isinstance(predictions, tuple) and len(predictions) > 0:
        scores = predictions[0][0]
        logger.debug("SASRec predictions extracted from tuple[0][0]")
    else:
        scores = predictions[0]
        logger.debug("SASRec predictions extracted from predictions[0]")

    logger.debug(f"SASRec scores shape: {scores.shape if hasattr(scores, 'shape') else 'N/A'}")
    # Get top-k items
    top_k_indices = np.argsort(-scores)[:num_recommendations*2]
    logger.debug(f"SASRec top_k_indices (before filtering): {top_k_indices}")
    
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
    logger.info(f"SASRec generated {len(recommendations)} recommendations: {recommendations}")
    return recommendations