import pytest
from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset





def test_cannot_instantiate_verification_module():
    """Ensure VerificationModule cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ExperimentDataset()
