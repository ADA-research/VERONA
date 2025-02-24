import pytest
import numpy as np
from pathlib import Path
from result import Err, Ok
from unittest.mock import MagicMock
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule, parse_counter_example, parse_counter_example_label
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator

from autoverify.verifier.verification_result import CompleteVerificationData

class MockVerifier:
    def verify_property(self, network_path, property_path, timeout, config=None):
        return Ok(CompleteVerificationData(result="SAT", counter_example="(X_0 0.1)\n(X_1 0.2)\n(Y_0 0.3)\n(Y_1 0.4)"))

class MockPropertyGenerator(PropertyGenerator):
    def create_vnnlib_property(self, image: np.array, image_class: int, epsilon: float) -> VNNLibProperty:
        return VNNLibProperty(name="test_property", content="test_content")

    def get_dict_for_epsilon_result(self) -> dict:
        return {}

    def to_dict(self) -> dict:
        return {}

    @classmethod
    def from_dict(cls, data: dict):
        return cls()

@pytest.fixture
def verifier():
    return MockVerifier()

@pytest.fixture
def property_generator():
    return MockPropertyGenerator()

@pytest.fixture
def verification_context(property_generator):
    class Network:
        def __init__(self):
            self.path = Path("/path/to/network")

    class DataPoint:
        def __init__(self):
            self.label = 1
            self.data = np.random.rand(28, 28)

    network = Network()
    data_point = DataPoint()
    tmp_path = Path("/tmp")
    return VerificationContext(network, data_point, tmp_path, property_generator)

@pytest.fixture
def auto_verify_module(verifier):
    return AutoVerifyModule(verifier, timeout=60.0)

def test_auto_verify_module_initialization(auto_verify_module, verifier):
    assert auto_verify_module.verifier == verifier
    assert auto_verify_module.timeout == 60.0
    assert auto_verify_module.config is None

def test_auto_verify_module_verify(auto_verify_module, verification_context):
    epsilon = 0.1
    result = auto_verify_module.verify(verification_context, epsilon)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == "SAT"

def test_parse_counter_example():
    result = Ok(CompleteVerificationData(result="SAT", counter_example="(X_0 0.1)\n(X_1 0.2)\n(Y_0 0.3)\n(Y_1 0.4)"))
    counter_example = parse_counter_example(result)
    assert isinstance(counter_example, np.ndarray)
    assert counter_example.shape == (28, 28)

def test_parse_counter_example_label():
    result = Ok(CompleteVerificationData(result="SAT", counter_example="(X_0 0.1)\n(X_1 0.2)\n(Y_0 0.3)\n(Y_1 0.4)"))
    label = parse_counter_example_label(result)
    assert isinstance(label, int)
    assert label == 1