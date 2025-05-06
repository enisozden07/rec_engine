"""Model handlers for different recommendation models."""

from .sasrec import SASRecHandler
from .ssept import SSEPTHandler
from .lightgcn import LightGCNHandler

def get_model_handler(model_type: str):
    """Get the appropriate model handler."""
    handlers = {
        "sasrec": SASRecHandler(),
        "ssept": SSEPTHandler(),
        "lightgcn": LightGCNHandler()
    }
    return handlers.get(model_type)