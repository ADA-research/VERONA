import pytest

from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PropertyGenerator()




