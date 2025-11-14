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
import pytest
import torch
import torch.nn as nn

from ada_verona.database.machine_learning_model.torch_model_wrapper import TorchModelWrapper


def test_torch_model_wrapper_initialization(torch_model_wrapper, mock_torch_model):

    assert torch_model_wrapper.torch_model == mock_torch_model
    assert torch_model_wrapper.input_shape == (2, 3)


def test_torch_model_wrapper_forward(torch_model_wrapper):
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    output = torch_model_wrapper(input_tensor)

    assert torch.allclose(output, torch.tensor([21.0]))


def test_torch_model_wrapper_reshape(torch_model_wrapper):
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    reshaped_tensor = input_tensor.reshape(torch_model_wrapper.input_shape)

    assert reshaped_tensor.shape == torch_model_wrapper.input_shape
    assert torch.equal(
        reshaped_tensor, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )


def test_torch_model_wrapper_with_tuple(mock_torch_model):

    input_shape = (2, 3)
    wrapper = TorchModelWrapper(torch_model=mock_torch_model, input_shape=input_shape)
    
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    output = wrapper(input_tensor)
    
    assert torch.allclose(output, torch.tensor([21.0]))
    
def test_torch_model_wrapper_with_numpy_array(mock_torch_model):
    
    input_shape = np.array([2, 3])
    wrapper = TorchModelWrapper(torch_model=mock_torch_model, input_shape=input_shape)
    
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    output = wrapper(input_tensor)
    
    assert torch.allclose(output, torch.tensor([21.0]))

def test_torch_model_wrapper_numpy_device_and_dtype():
    torch_model = nn.Linear(3, 1)
    wrapper = TorchModelWrapper(torch_model, (2, 3))
  
    np_input = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
    output = wrapper(np_input)

    device = next(torch_model.parameters()).device
    assert output.device == device
    assert output.dtype == torch.float32
    assert output.shape[0] == 2
    
def test_torch_model_wrapper_different_shape(mock_torch_model):
    wrapper = TorchModelWrapper(mock_torch_model, (6,))
    input_tensor = torch.tensor([[1., 2., 3., 4., 5., 6.]])
    
    output = wrapper(input_tensor)
    assert torch.allclose(output, torch.tensor([21.0]))
    
    
def test_torch_model_wrapper_invalid_shape(mock_torch_model):
    wrapper = TorchModelWrapper(mock_torch_model, (3, 3))
    bad_input = torch.tensor([[1., 2., 3., 4., 5., 6.]])
    
    with pytest.raises(RuntimeError):
        _ = wrapper(bad_input)
        
def test_torch_model_wrapper_already_reshaped(mock_torch_model):
    wrapper = TorchModelWrapper(mock_torch_model, (2, 3))
    input_tensor = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    
    output = wrapper(input_tensor)
    assert torch.allclose(output, torch.tensor([21.0]))