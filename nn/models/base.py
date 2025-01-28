"""
Base implementation for all models in the `nn.models` module
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

    def __repr__(self) -> str:
        """
        String representation of the `Model` class
        """
        return f"{self.__class__.__name__}()"
