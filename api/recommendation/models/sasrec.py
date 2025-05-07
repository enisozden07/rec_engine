"""SASRec model implementation."""

import os
import json
import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import random

from .base import ModelHandler
from recommenders.models.sasrec.model import SASREC

logger = logging.getLogger(__name__)

class DummySASREC:
    """Dummy implementation of SASREC model for testing and development."""
    
    def __init__(self, config):
        """Initialize the dummy model with the same config structure as the real model."""
        self.config = config
        self.seq_max_len = config.get("seq_max_len", 50)
        # Extract item_num from config or set to the max value from item mapping (1034)
        self.item_num = config.get("item_num", 1034)
        self.hidden_units = config.get("hidden_units", 128)
        self.item_embedding_size = config.get("item_embedding_size", 100)
        logger.info(f"Initialized DummySASREC with seq_max_len={self.seq_max_len}, item_num={self.item_num}")
        
        # Create a lookup table for similarity scoring to make recommendations more realistic
        # This makes items closer in ID more likely to be recommended together
        self.item_similarities = {}
        for i in range(1, self.item_num + 1):
            # Create a "neighborhood" of similar items
            similar_items = set()
            # Items with similar IDs tend to be more alike
            for j in range(max(1, i-10), min(self.item_num + 1, i+11)):
                if i != j:
                    similar_items.add(j)
            # Add some random items as somewhat similar
            for _ in range(20):
                similar_items.add(random.randint(1, self.item_num))
            self.item_similarities[i] = list(similar_items)
        
    def load(self, checkpoint_path):
        """Dummy load method that does nothing but log."""
        logger.info(f"DummySASREC pretending to load checkpoint from: {checkpoint_path}")
        return self
    
    def predict(self, inputs_dict):
        """
        Generate dummy predictions based on input sequence.
        Makes predictions somewhat intelligent by giving higher scores to items similar to the input.
        """
        input_seq = inputs_dict.get("input_seq", None)
        candidate = inputs_dict.get("candidate", None)
        
        if input_seq is None or candidate is None:
            raise KeyError("Input_seq or candidate missing from inputs_dict")
        
        # Convert tensors to numpy if needed
        if hasattr(input_seq, 'numpy'):
            input_seq = input_seq.numpy()
        if hasattr(candidate, 'numpy'):
            candidate = candidate.numpy()
        
        # Extract non-zero items from the input sequence
        batch_size = input_seq.shape[0]
        sequence_items = set()
        for b in range(batch_size):
            for item_id in input_seq[b]:
                if item_id > 0:  # Skip padding (0)
                    sequence_items.add(int(item_id))
        
        # Generate scores for candidate items
        # Shape will be [batch_size, number_of_candidates]
        scores = np.zeros((batch_size, candidate.shape[1]), dtype=np.float32)
        
        for b in range(batch_size):
            for i, item_id in enumerate(candidate[b]):
                item_id = int(item_id)
                # Base score is a random value
                base_score = 0.1 + 0.8 * random.random()
                
                # Boost score if item is similar to items in the sequence
                similarity_boost = 0.0
                for seq_item in sequence_items:
                    if item_id in self.item_similarities.get(seq_item, []):
                        similarity_boost += 0.2 * random.random()
                
                # Some popular items (lower IDs often) get a boost
                popularity_boost = max(0, 0.3 * (1.0 - item_id / self.item_num))
                
                # Combine scores with some randomness
                scores[b, i] = base_score + similarity_boost + popularity_boost
        
        return tf.constant(scores, dtype=tf.float32)


class SASRecHandler(ModelHandler):
    """Handler for SASRec model."""
    
    def load_model_files(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Load SASRec model files from disk (or create dummy model)."""
        logger.info(f"Loading SASRec model files from model_path: {model_path}, model_type: {model_type}")
        result = {}
        
        # Load configuration
        config_path = os.path.join(model_path, model_type, "configs", f"{model_type}_config.json")
        logger.info(f"SASRec config path: {config_path}")
        
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.debug(f"SASRec config loaded: {config}")
        except FileNotFoundError:
            # If config file doesn't exist, create a default config
            logger.warning(f"Config file not found at {config_path}, using default config")
            config = {
                "seq_max_len": 50,
                "item_num": 1034,  # Maximum item ID from the mapping
                "hidden_units": 128,
                "item_embedding_size": 100,
                "num_blocks": 2,
                "num_heads": 2,
                "dropout_rate": 0.2,
                "l2_emb": 0.0,
                "device": "cpu"
            }
        
        # Create dummy model
        model = DummySASREC(config)
        
        # Checkpoint path (for logging purposes only with dummy model)
        checkpoint_path = os.path.join(model_path, model_type, "checkpoints", f"{model_type}.ckpt")
        logger.info(f"SASRec checkpoint path (not actually loaded): {checkpoint_path}")
        
        # Pretend to load, but it does nothing for the dummy model
        model.load(checkpoint_path)
        logger.info("Using dummy SASRec model (no actual checkpoint loaded)")
        
        # Store the model and config
        result["model"] = model
        result["config"] = config
        
        return result
    
    def predict(self, model, inputs_dict):
        """Make predictions with the loaded model using its own predict method."""
        # 'inputs_dict' should be a dictionary like {'input_seq': tensor, 'candidate': tensor}
        logger.debug(f"SASRec predict called with inputs_dict keys: {inputs_dict.keys()}")
        try:
            # Works the same for both real and dummy model
            return model.predict(inputs_dict)
        except KeyError as ke:
            logger.error(f"KeyError directly within SASRecHandler.predict during model.predict: {ke}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Generic exception directly within SASRecHandler.predict during model.predict: {e}", exc_info=True)
            raise


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
    
    seq_max_len = getattr(model, "seq_max_len", 50)
    item_num = getattr(model, "item_num", None) # Get item_num from the model instance

    if item_num is None:
        logger.error("SASRec model 'item_num' attribute not found. Cannot generate candidates.")
        return []
        
    logger.debug(f"SASRec model seq_max_len: {seq_max_len}, item_num: {item_num}")
    
    seq = np.zeros([1, seq_max_len], dtype=np.int32)
    idx = min(len(basket_indices), seq_max_len)
    seq[0, -idx:] = basket_indices[-idx:]
    
    seq_tensor = tf.constant(seq, dtype=tf.int32)
    
    # Prepare candidate items: all items in the catalog
    # Item indices are typically 1-based in SASRec data preparation, up to item_num.
    # The embedding layer is often item_num + 1 to handle a padding/masking index 0.
    # Ensure candidate items are within the valid range for item_embedding_layer.
    all_items = np.arange(1, item_num + 1, dtype=np.int32) # Assuming items are 1 to item_num
    candidate_items_tensor = tf.constant([all_items]) # Shape: (1, item_num)

    logger.debug(f"SASRec input seq_tensor: {seq_tensor.shape}, candidate_items_tensor: {candidate_items_tensor.shape}")
    
    model_handler = SASRecHandler()
    
    # The model.predict method expects a dictionary
    prediction_inputs = {
        "input_seq": seq_tensor,
        "candidate": candidate_items_tensor
    }
    
    # predictions are logits for each candidate item
    predictions_output = model_handler.predict(model, prediction_inputs) # This will be a tensor of logits
    logger.debug(f"SASRec raw predictions_output shape: {predictions_output.shape if hasattr(predictions_output, 'shape') else 'N/A'}, type: {type(predictions_output)}")

    # Ensure predictions_output is a NumPy array for processing
    if hasattr(predictions_output, 'numpy'):
        scores = predictions_output.numpy().flatten() # Flatten to 1D array of scores
    else:
        scores = np.array(predictions_output).flatten()

    logger.debug(f"SASRec scores shape after flatten: {scores.shape if hasattr(scores, 'shape') else 'N/A'}")

    # Get top-k items. Scores correspond to all_items.
    # We need to sort and get indices relative to `all_items`
    # Add a check to ensure scores length matches all_items length
    if len(scores) != len(all_items):
        logger.error(f"SASRec scores length ({len(scores)}) does not match candidate items length ({len(all_items)}). Cannot proceed.")
        return []

    # Argsort returns indices that would sort the array.
    # We want the items with the highest scores, so we sort in descending order.
    # Slicing with [::-1] reverses the sorted indices for descending order.
    sorted_candidate_indices = np.argsort(scores)[::-1] 
    
    recommendations = []
    # Filter out items already in the basket from recommendations
    basket_set = set(basket_indices)

    for i in range(len(sorted_candidate_indices)):
        if len(recommendations) >= num_recommendations:
            break
        
        original_item_idx_in_candidates = sorted_candidate_indices[i]
        # The actual item ID is all_items[original_item_idx_in_candidates]
        # This is because scores are aligned with all_items
        recommended_item_id = all_items[original_item_idx_in_candidates]

        if recommended_item_id not in basket_set: # Exclude items already in basket
            recommendations.append({
                "index": int(recommended_item_id), # Store the actual item ID (index)
                "score": float(scores[original_item_idx_in_candidates]) # Score for this item
            })
            
    logger.info(f"SASRec generated {len(recommendations)} recommendations: {recommendations}")
    return recommendations