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