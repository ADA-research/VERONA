import pytest

from ada_verona.verification_module.attacks.attack import Attack


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Attack()
