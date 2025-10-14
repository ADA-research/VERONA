import pytest

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset


def test_cannot_instantiate_dataset():
    """Ensure ExperimentDataset cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ExperimentDataset()


def test_abstract_methods_raise_not_implemented_error():
    # Call the abstract methods on the class itself (unbound)
    with pytest.raises(NotImplementedError):
        ExperimentDataset.__len__(ExperimentDataset)
        
    with pytest.raises(NotImplementedError):
        ExperimentDataset.__getitem__(ExperimentDataset, 0)
        
    with pytest.raises(NotImplementedError):
        ExperimentDataset.get_subset(ExperimentDataset, [0])