import pytest

from robustness_experiment_box.dataset_sampler.dataset_sampler import DatasetSampler


def test_cannot_instantiate_datasetsampler():
    """Ensure VerificationModule cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DatasetSampler()
