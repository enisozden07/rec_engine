"""Utility functions for the recommendation engine."""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

def load_id_mappings(mappings_dir: str) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """
    Load user and item ID mappings from CSV files.
    
    Args:
        mappings_dir: Directory containing mapping CSV files
        
    Returns:
        Tuple of (item_id_to_idx, idx_to_item_id, user_id_to_idx, idx_to_user_id) dictionaries
    """
    item_id_to_idx = {}
    idx_to_item_id = {}
    user_id_to_idx = {}
    idx_to_user_id = {}
    
    try:
        # Load user mapping
        user_map_path = os.path.join(mappings_dir, "user_mapping.csv")
        if os.path.exists(user_map_path):
            user_df = pd.read_csv(user_map_path)
            if len(user_df.columns) >= 2:
                # Assuming first column is original ID and second is index
                orig_id_col = user_df.columns[0]
                idx_col = user_df.columns[1]
                
                # Create mappings in both directions
                for _, row in user_df.iterrows():
                    original_id = str(row[orig_id_col])
                    idx = int(row[idx_col])
                    user_id_to_idx[original_id] = idx
                    idx_to_user_id[idx] = original_id
                
                logger.info(f"Loaded {len(user_id_to_idx)} user mappings from {user_map_path}")
        
        # Load item mapping
        item_map_path = os.path.join(mappings_dir, "item_mappings.csv")
        if not os.path.exists(item_map_path):
            item_map_path = os.path.join(mappings_dir, "item_mapping.csv")  # Try alternative filename
            
        if os.path.exists(item_map_path):
            item_df = pd.read_csv(item_map_path)
            if len(item_df.columns) >= 2:
                # Assuming first column is original ID and second is index
                orig_id_col = item_df.columns[0]
                idx_col = item_df.columns[1]
                
                # Create mappings in both directions
                for _, row in item_df.iterrows():
                    original_id = str(row[orig_id_col])
                    idx = int(row[idx_col])
                    item_id_to_idx[original_id] = idx
                    idx_to_item_id[idx] = original_id
                
                logger.info(f"Loaded {len(item_id_to_idx)} item mappings from {item_map_path}")
                
    except Exception as e:
        logger.error(f"Failed to load mappings from CSV: {str(e)}")
    
    return item_id_to_idx, idx_to_item_id, user_id_to_idx, idx_to_user_id

def map_user_id(user_id: str, user_id_to_idx: Dict[str, int]) -> Optional[int]:
    """
    Map external user ID to internal index.
    
    Args:
        user_id: External user ID to map
        user_id_to_idx: Mapping dictionary from user IDs to indices
        
    Returns:
        Mapped index or None if not found
    """
    if not user_id:
        return None
        
    # Try different formats
    for id_format in [user_id, str(user_id), int(user_id) if user_id.isdigit() else None]:
        if id_format is not None and str(id_format) in user_id_to_idx:
            return user_id_to_idx[str(id_format)]
            
    logger.warning(f"User ID {user_id} not found in mappings")
    return None
    
def map_item_ids(item_ids: List[str], item_id_to_idx: Dict[str, int]) -> List[int]:
    """
    Map external item IDs to internal indices.
    
    Args:
        item_ids: List of external item IDs to map
        item_id_to_idx: Mapping dictionary from item IDs to indices
        
    Returns:
        List of mapped indices (only includes successfully mapped items)
    """
    mapped_ids = []
    for item_id in item_ids:
        # Try different formats
        for id_format in [item_id, str(item_id), int(item_id) if item_id.isdigit() else None]:
            if id_format is not None and str(id_format) in item_id_to_idx:
                mapped_ids.append(item_id_to_idx[str(id_format)])
                break
    
    return mapped_ids

def map_recommendations_to_original_ids(
    recommendations: List[Dict[str, Any]], 
    idx_to_item_id: Dict[int, str]
) -> List[Dict[str, Any]]:
    """
    Convert model recommendations with internal indices to original product IDs.
    
    Args:
        recommendations: List of recommendation dictionaries with "index" and "score" fields
        idx_to_item_id: Mapping dictionary from indices to original item IDs
        
    Returns:
        List of recommendation dictionaries with "product_id", "score", etc. fields
    """
    mapped_recommendations = []
    for rec in recommendations:
        try:
            # Ensure we have an integer index
            idx = int(rec["index"])
            # Look up the original ID in the mapping
            if idx in idx_to_item_id:
                original_id = idx_to_item_id[idx]
                mapped_recommendations.append({
                    "product_id": original_id,
                    "score": float(rec["score"]),
                    "product_name": f"Product {original_id}",
                    "category": rec.get("category", "unknown")
                })
            else:
                # If index not found in mapping, log a warning and include it with the index as product ID
                logger.warning(f"Item index {idx} not found in reverse mapping")
                mapped_recommendations.append({
                    "product_id": str(idx),
                    "score": float(rec["score"]),
                    "product_name": f"Product {idx} (unmapped)",
                    "category": "unknown"
                })
        except (ValueError, KeyError) as e:
            logger.warning(f"Error mapping recommendation {rec}: {e}")
            # Include this recommendation with a warning note
            if 'index' in rec and 'score' in rec:
                mapped_recommendations.append({
                    "product_id": str(rec['index']),
                    "score": float(rec["score"]),
                    "product_name": f"Error: {str(e)}",
                    "category": "unknown"
                })
    
    return mapped_recommendations