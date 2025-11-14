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
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
      


    @abstractmethod
    def get_input_shape(self) -> np.ndarray | tuple[int, ...]:
        """
        Get the input shape of the model.

        Returns:
            Union[np.ndarray, tuple[int, ...]]: The input shape of the model.
        """
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
      


    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert the network to a dictionary.

        Returns:
            dict: The dictionary representation of the network.
        """
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
      


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
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")


    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the network.

        Returns:
            str: The name of the network.
        """
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
    
    @property
    @abstractmethod
    def path(self) -> Path:
        """
        Get the path of the network.

        Returns:
            Path: The path of the network.
        """
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")

    @classmethod
    def from_file(cls, file: dict[Path]):
        """Create network from file
        Args: 
            file (dict[Path]): contains the paths to the relevant weights (for ONNX) 
            and additionally to the architecture file for PyTorch networks.
        
        Returns: 
            Created network from the correct class OR error. 
        """
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
        