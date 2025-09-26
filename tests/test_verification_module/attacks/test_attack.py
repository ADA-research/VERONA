import pytest

from ada_verona.verification_module.attacks.attack import Attack


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Attack()


def test_fuctions_raises_notimplementederror():
    # Call the unbound method so the body executes
    with pytest.raises(NotImplementedError):
        Attack.execute(Attack, None, 0,0,0)
