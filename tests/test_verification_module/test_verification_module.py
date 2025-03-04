import pytest

from robustness_experiment_box.verification_module.verification_module import VerificationModule


def test_cannot_instantiate_verification_module():
    """Ensure VerificationModule cannot be instantiated directly."""
    with pytest.raises(TypeError):
        VerificationModule()
