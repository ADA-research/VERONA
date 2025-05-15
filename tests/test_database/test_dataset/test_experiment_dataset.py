import pytest
from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset
from robustness_experiment_box.database.dataset.data_point import DataPoint


def test_cannot_instantiate_verification_module():
    """Ensure VerificationModule cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ExperimentDataset()
class MockExperimentDataset(ExperimentDataset):
    def __init__(self, data):
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> DataPoint:
        return self._data[idx]

    def get_subset(self, indices: list[int]) -> 'MockExperimentDataset':
        subset = [self._data[i] for i in indices]
        return MockExperimentDataset(subset)


@pytest.fixture
def sample_data():
    return [DataPoint(id=i, data=[0], label=i % 2) for i in range(5)]


@pytest.fixture
def mock_dataset(sample_data):
    return MockExperimentDataset(sample_data)


def test_len(mock_dataset):
    assert len(mock_dataset) == 5


def test_get_item(mock_dataset):
    item = mock_dataset[0]
    assert isinstance(item, DataPoint)
    assert item.id == 0
    assert item.label == 0


def test_get_subset(mock_dataset):
    subset = mock_dataset.get_subset([1, 3])
    assert isinstance(subset, MockExperimentDataset)
    assert len(subset) == 2
    assert subset[0].id == 1
    assert subset[1].id == 3