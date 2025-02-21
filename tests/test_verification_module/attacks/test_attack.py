from abc import ABC, abstractmethod
from robustness_experiment_box.verification_module.attacks.attack import Attack
import pytest


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Attack()
