"""Model handlers for different recommendation models."""

from .sasrec import SASRecHandler
from .ssept import SSEPTHandler
from .lightgcn import LightGCNHandler

def get_model_handler(model_type: str):
    """Get the appropriate model handler based on the model type."""
    handlers = {
        "sasrec": SASRecHandler(),
        "ssept": SSEPTHandler(),
        "lightgcn": LightGCNHandler()
    }
    
    if model_type not in handlers:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return handlers[model_type]