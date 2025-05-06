"""Utility functions for the recommendation engine."""

from typing import Dict, Any

def create_dummy_mappings() -> Dict[str, Dict[str, Any]]:
    """Create dummy mappings for testing."""
    return {
        "item_map": {"P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5},
        "reverse_item_map": {1: "P1", 2: "P2", 3: "P3", 4: "P4", 5: "P5"}
    }