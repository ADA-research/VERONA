from pathlib import Path

import numpy as np
import onnx
import torch
from onnx2torch import convert

from robustness_experiment_box.database.machine_learning_method.network import Network
from robustness_experiment_box.database.machine_learning_method.torch_model_wrapper import TorchModelWrapper


class ONNXNetwork(Network):
    """
    Data class representing an ONNX network with its path.

    Attributes:
        path (Path): The path to the network file.
        onnx_model (onnx.ModelProto, optional): The loaded ONNX model. Defaults to None.
        torch_model_wrapper (TorchModelWrapper, optional): The PyTorch model wrapper. Defaults to None.
    """

    def __init__(self, path: Path) -> None:
        """
        Initialize the Network with the given path.

        Args:
            path (Path): The path to the network file.
        """
        self.path = path
        self.onnx_model = None
        self.torch_model_wrapper = None

    @property
    def name(self) -> str:
        """
        Get the name of the network.

        Returns:
            str: The name of the network.
        """
        return self.path.stem

    def load_onnx_model(self) -> onnx.ModelProto:
        """
        Load the ONNX model from the network path.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        model = self.onnx_model
        if model is None:
            model = onnx.load(str(self.path))
            self.onnx_model = model

        return model

    def get_input_shape(self) -> np.ndarray:
        """
        Get the input shape of the ONNX model.

        Returns:
            np.ndarray: The input shape of the ONNX model.
        """
        model = self.load_onnx_model()
        input_shape = tuple([d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim])
        input_shape = [x if x != 0 else -1 for x in input_shape]

        return input_shape

    def load_pytorch_model(self) -> torch.nn.Module:
        """
        Load the PyTorch model from the ONNX model.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model_wrapper = self.torch_model_wrapper
        if torch_model_wrapper is None:
            torch_model = convert(self.path).to(device)
            torch_model_wrapper = TorchModelWrapper(torch_model, self.get_input_shape())
            self.torch_model_wrapper = torch_model_wrapper

        return torch_model_wrapper

    def to_dict(self) -> dict:
        """
        Convert the Network to a dictionary.

        Returns:
            dict: The dictionary representation of the Network.
        """
        
        return dict(network_path =  str(self.path), 
                type=self.__class__.__name__,
                module=self.__class__.__module__,
                    )

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a Network from a dictionary.

        Args:
            data (dict): The dictionary containing the Network attributes.

        Returns:
            Network: The created Network.
        """
        return cls(path=Path(data["network_path"]))
    
    @classmethod
    def from_file(cls, file:Path):
        """
        Create an ONNXNetwork from a dictionary.

        Args:
            file (Path): Path at which the network is stored. 

        Returns:
            ONNXNetwork: The created ONNXNetwork.
        """
        return cls(path = file)