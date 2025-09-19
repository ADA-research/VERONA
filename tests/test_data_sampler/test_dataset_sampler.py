import pytest

from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler


def test_cannot_instantiate_datasetsampler():
    """Ensure VerificationModule cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DatasetSampler()
