from abc import ABC, abstractmethod
from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator
import pytest


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PropertyGenerator()
