"""Base interfaces for recommendation models."""

from abc import ABC, abstractmethod
from typing import Dict, Any

class ModelHandler(ABC):
    """Base class for model handlers."""
    
    @abstractmethod
    def load_model_files(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Load model files from disk."""
        pass