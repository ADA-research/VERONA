import pytest
from result import Ok
from torch import load

from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.database.datastructure.onnx_network import ONNX_Network
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.verification_result import CompleteVerificationData
from robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from tests.test_verification_module.helpers.test_verification_module import TestVerificationModule


@pytest.fixture
def network():
    return Network("./tests/test_experiment/data/networks/mnist-net_256x2.onnx")


@pytest.fixture
def datapoint():
    return DataPoint(label=5, data=load("./tests/test_experiment/data/images/mnist_train_0.pt"), id="0")


@pytest.fixture
def attack():
    return TestVerificationModule()

@pytest.fixture
def attack_estimation_module(attack):
    return AttackEstimationModule(attack)

@pytest.fixture
def verification_context(network, datapoint, tmp_path):
    property_generator = One2AnyPropertyGenerator()
    return VerificationContext(network, datapoint, tmp_path, property_generator)
@pytest.fixture
def verifier():
    return TestVerificationModule()

@pytest.fixture
def result(datapoint):
    # Flatten the tensor and format each value
    formatted_strings = [f"(X_{i} {datapoint.data.flatten()[i]:.4f})" for i in range(28 * 28)]

    # Join all entries with newlines
    result = "\n".join(formatted_strings)
    result += "\n(Y_0 0.3)"

    return Ok(CompleteVerificationData(result="SAT", counter_example=result, took =10))
