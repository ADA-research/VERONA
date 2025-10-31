from pathlib import Path

import numpy as np
import torch

from ada_verona.database.machine_learning_model.network import Network
from ada_verona.database.machine_learning_model.torch_model_wrapper import TorchModelWrapper


class PyTorchNetwork(Network):
    """
    A class representing a PyTorch network with architecture and weights files.

    Attributes:
        model (torch.nn.Module, optional): The loaded PyTorch model. Defaults to None.
        torch_model_wrapper (TorchModelWrapper, optional): The PyTorch model wrapper. Defaults to None.
        name: A chosen name for the model.
        input_shape (tuple[int]): Input shape of the model.
    """

    def __init__(self, model: torch.nn.Module, input_shape: tuple[int], name: str) -> None:
        """
        Initialize the PyTorchNetwork with architecture and weights paths.

        Args:
            model (torch.nn.Module, optional): The loaded PyTorch model. Defaults to None.
            input_shape (tuple[int]): Input shape of the model.
            name: A chosen name for the model.
        """

        self.model = model
        self.input_shape = input_shape
        self._name = name
        self.torch_model_wrapper = None

    @property
    def name(self) -> str:
        """
        Get the name of the network.

        Returns:
            str: The name of the network.
        """
        return self._name
    
    @property
    def path(self) -> Path:
        """
        Get the path of the network.

        Returns:
            Path: The path of the network.
        """
        return None
    
    def get_input_shape(self) -> np.ndarray:
        """
        Get the input shape of the PyTorch model.

        Returns:
            np.ndarray: the input_shape
        """

        return self.input_shape

    def load_pytorch_model(self) -> torch.nn.Module:
        """
        Load the PyTorch model and wrap it in a TorchModelWrapper.

        Returns:
            torch.nn.Module: The wrapped PyTorch model.
        """
        if self.torch_model_wrapper is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = self.model.to(device)
            model.eval()
            
            self.torch_model_wrapper = TorchModelWrapper(model, self.get_input_shape())
        
        return self.torch_model_wrapper
    

    def to_dict(self):
        raise NotImplementedError("PytorchNetwork does not support to_dict() function currently.")
    
    def from_dict(cls, data: dict):
         raise NotImplementedError("PytorchNetwork does not support from_dict() function currently.")