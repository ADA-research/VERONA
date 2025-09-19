import importlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class Network(ABC):
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
    def get_input_shape(self) -> np.ndarray | tuple[int, ...]:
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
    def from_dict(cls, data: dict) -> "Network":
        """
        Create a network from a dictionary.

        Args:
            data (dict): The dictionary containing the network attributes.

        Returns:
            BaseNetwork: The created network.
        """
        class_name = data.pop("type", None)
        module_name = data.pop("module", None)  # Get module info
        if not class_name or not module_name:
            raise ValueError("Missing 'class' or 'module' key in dictionary")

        try:
            module = importlib.import_module(module_name)  # Dynamically import module
            subclass = getattr(module, class_name)  # Get class from module
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Could not import {class_name} from {module_name}: {e}") from e

        return subclass.from_dict(data)  # Call subclass's `from_dict`


    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the network.

        Returns:
            str: The name of the network.
        """
        pass
    

    @classmethod
    def from_file(cls, file:Path):

        if file.suffix == ".onnx":
            module = importlib.import_module("robustness_experiment_box.database.machine_learning_method.onnx_network")
            subclass = module.ONNXNetwork  # Get class from module
        else:
            raise NotImplementedError(f"Only .onnx files are supported at the moment, got: {file.suffix}")
        
        return subclass.from_file(file)