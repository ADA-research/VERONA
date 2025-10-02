import pytest

from ada_verona.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator


def test_cannot_instantiate_epsilon_valu_estimator():
    """Ensure Network cannot be instantiated directly."""
    with pytest.raises(TypeError):
        EpsilonValueEstimator()


def test_abstract_methods_raise_notimplementederror():
    # Call the abstract methods on the class itself (unbound)
    with pytest.raises(NotImplementedError):
        EpsilonValueEstimator.compute_epsilon_value(EpsilonValueEstimator, None)


