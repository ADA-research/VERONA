"""
Script defining a network object 
"""
import onnx
import torch
import numpy as np
from onnx2torch import convert
from pathlib import Path
from dataclasses import dataclass

from robustness_experiment_box.database.torch_model_wrapper import TorchModelWrapper

class Network:
    """
    Data class representing a network with its path.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.onnx_model = None
        self.torch_model_wrapper = None

    def load_onnx_model(self) -> onnx.ModelProto:

        model = self.onnx_model
        if model is None:
            model = onnx.load(str(self.path))
            self.onnx_model = model

        return model
    
    def get_input_shape(self) -> np.ndarray:
        model = self.load_onnx_model()
        input_shape = tuple([d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim])
        input_shape = [x if x != 0 else -1 for x in input_shape ]

        return input_shape
    
    def load_pytorch_model(self) -> torch.nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model_wrapper = self.torch_model_wrapper
        if torch_model_wrapper is None:
            torch_model = convert(self.path).to(device)
            torch_model_wrapper = TorchModelWrapper(torch_model, self.get_input_shape())
            self.torch_model_wrapper = torch_model_wrapper

        return torch_model_wrapper
    
    def to_dict(self):
        return {
            'network_path': str(self.path)
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(path = Path(data['network_path']))