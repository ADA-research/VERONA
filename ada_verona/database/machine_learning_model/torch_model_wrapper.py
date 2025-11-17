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

import numpy as np
import torch


class TorchModelWrapper(torch.nn.Module):
    """
    A wrapper class for a PyTorch model to reshape the input before passing it to the model.
    """

    def __init__(self, torch_model: torch.nn.Module, input_shape: tuple[int]):
        """
        Initialize the TorchModelWrapper with the given PyTorch model and input shape.

        Args:
            torch_model (torch.nn.Module): The PyTorch model to wrap.
            input_shape: The input shape to reshape the input tensor. Can be tuple[int] or np.ndarray.
        """
        super().__init__()
        self.torch_model = torch_model
        self.input_shape = input_shape

    def forward(self, x):
        """
        Forward pass of the TorchModelWrapper.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor from the wrapped PyTorch model.
        """
        
        if isinstance(x, np.ndarray):
        # ensure correct dtype/device if needed
            x = torch.from_numpy(x).to(
                dtype=torch.float32,
                device=next(self.torch_model.parameters()).device
            )
            x = x.reshape(*self.input_shape)  # tuple unpacking

        else:
        # Assume it's already a torch.Tensor
            x = x.reshape(*self.input_shape)

        x = self.torch_model(x)
        return x
