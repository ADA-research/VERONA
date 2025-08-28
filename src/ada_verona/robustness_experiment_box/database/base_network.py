from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class BaseNetwork(ABC):
    """
    Abstract base class for networks that can be either ONNX or PyTorch.

    This class provides a common interface for both network types.
    """

    @abstractmethod
    def load_pytorch_model(self) -> torch.nn.Module:
        """
        Load the PyTorch model.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        pass

    @abstractmethod
    def get_input_shape(self) -> Union[np.ndarray, tuple[int, ...]]:
        """
        Get the input shape of the model.

        Returns:
            Union[np.ndarray, tuple[int, ...]]: The input shape of the model.
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert the network to a dictionary.

        Returns:
            dict: The dictionary representation of the network.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "BaseNetwork":
        """
        Create a network from a dictionary.

        Args:
            data (dict): The dictionary containing the network attributes.

        Returns:
            BaseNetwork: The created network.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the network.

        Returns:
            str: The name of the network.
        """
        pass
