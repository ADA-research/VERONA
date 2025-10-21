import pytest
from result import Ok
from torch import load

from ada_verona.database.dataset.data_point import DataPoint
from ada_verona.database.machine_learning_model.onnx_network import ONNXNetwork
from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData
from ada_verona.verification_module.attack_estimation_module import AttackEstimationModule
from ada_verona.verification_module.auto_verify_module import AutoVerifyModule
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.verification_module.test_verification_module import TestVerificationModule


@pytest.fixture
def network():
    return ONNXNetwork("examples/example_experiment/data/networks/mnist-net_256x2.onnx")


@pytest.fixture
def datapoint():
    return DataPoint(label=5, data=load("examples/example_experiment/data/images/mnist_train_0.pt"), id="0")


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
def auto_verify_module(verifier):
    return AutoVerifyModule(verifier, timeout=60)

@pytest.fixture
def auto_verify_module_config(verifier, tmp_path):
    config = tmp_path / "dummy_config.yaml"
    config.write_text("timeout: 60\noption: test")
    return AutoVerifyModule(verifier, timeout=60, config=config)

    
@pytest.fixture
def result(datapoint):
    formatted_strings = [f"(X_{i} {datapoint.data.flatten()[i]:.4f})" for i in range(28 * 28)]

    result = "\n".join(formatted_strings)
    result += "\n(Y_0 0.3)"

    return Ok(CompleteVerificationData(result="SAT", counter_example=result, took =10))

