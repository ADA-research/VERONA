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

import pytest
import torch
from torch.utils.data import Dataset

from ada_verona.database.dataset.data_point import DataPoint
from ada_verona.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset


class MockDataset(Dataset):
    """
    A mock PyTorch dataset for testing purposes.
    """

    def __init__(self):
        self.data = [
            (torch.tensor([1.0, 2.0, 3.0]), 0),
            (torch.tensor([4.0, 5.0, 6.0]), 1),
            (torch.tensor([7.0, 8.0, 9.0]), 2),
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def mock_pytorch_experiment_dataset():
    dataset = MockDataset()
    return PytorchExperimentDataset(dataset)


def test_getitem(mock_pytorch_experiment_dataset):

    data_point = mock_pytorch_experiment_dataset[1]


    assert isinstance(data_point, DataPoint)
    assert data_point.id == 1
    assert data_point.label == 1
    assert torch.equal(data_point.data, torch.tensor([4.0, 5.0, 6.0]))


def test_get_subset(mock_pytorch_experiment_dataset):

    subset = mock_pytorch_experiment_dataset.get_subset([0, 2])


    assert len(subset) == 2
    assert subset[0].id == 0
    assert subset[1].id == 2
    assert torch.equal(subset[0].data, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(subset[1].data, torch.tensor([7.0, 8.0, 9.0]))


def test_str(mock_pytorch_experiment_dataset):

    dataset_str = str(mock_pytorch_experiment_dataset)


    assert dataset_str == "[0, 1, 2]"