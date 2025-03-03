from robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
from robustness_experiment_box.verification_module.test_verification_module import TestVerificationModule
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.verification_module.nnenum_module import NnenumModule
from torch import nn, load
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule
import pytest
from result import Err, Ok
from autoverify.verifier.verification_result import CompleteVerificationData


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
def nnenum_module():
    return NnenumModule(timeout=60.0)


@pytest.fixture
def verifier():
    return TestVerificationModule()

@pytest.fixture
def auto_verify_module(verifier):
    return AutoVerifyModule(verifier, timeout=60)


@pytest.fixture
def result(datapoint):
    data = """
   (X_3 0.1)
   (Y_0 0.2)
    """
    return Ok(CompleteVerificationData(result="SAT", counter_example=data, took =10))
