import pytest

from ada_verona.verification_module.attacks.attack import Attack


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Attack()

def test_abstract_methods_raise_not_implemented_error():
    # Call the abstract methods on the class itself (unbound)
    with pytest.raises(NotImplementedError):
        Attack.execute(Attack,None,None,None,None)
    