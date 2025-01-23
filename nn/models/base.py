"""
Base implementation for all models
"""

from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """
    Abstract base class for model implementation
    """

    @abstractmethod
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of model
        """
        pass

    @abstractmethod
    def backward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Backward pass of model
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run model inference on input
        """
        pass
