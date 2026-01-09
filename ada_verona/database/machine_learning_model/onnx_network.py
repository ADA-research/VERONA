# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx2torch import convert
from onnxsim import simplify

from ada_verona.database.machine_learning_model.network import Network
from ada_verona.database.machine_learning_model.torch_model_wrapper import TorchModelWrapper


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
        self._path = path
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
    
    @property
    def path(self) -> Path:
        """
        Get the path of the network.

        Returns:
            Path: The path of the network.
        """
        return self._path

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
            onnx_model = self.load_onnx_model()
            # Simplify model
            try:
                model_simp, check = simplify(onnx_model)
                if not check:
                    print(f"ONNX-simplifier validation failed for {self.name}, using original.")
                    model_to_convert = onnx_model
                else:
                    model_to_convert = model_simp
            except Exception as e:
                print(f"Simplification failed ({e}). Attempting to convert original model.")
                model_to_convert = onnx_model

            torch_model = convert(model_to_convert).to(device)
            
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
    def from_dict(cls, data: dict)-> "ONNXNetwork":
        """
        Create a Network from a dictionary.

        Args:
            data (dict): The dictionary containing the Network attributes.

        Returns:
            Network: The created Network.
        """
        return cls(path = data['network_path'])
    
    @classmethod
    def from_file(cls, file:Path)-> "ONNXNetwork":
        """
        Create a ONNXNetwork from a dictionary.

        Args:
            file (Path): Path at which the network is stored. 

        Returns:
            ONNXNetwork: The created ONNXNetwork.
        """

        if not file.is_file():
            raise FileNotFoundError(f"File not found: {file}")

        return cls(path = file)