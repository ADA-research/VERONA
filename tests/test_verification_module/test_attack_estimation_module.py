import pytest
import torch
from torch import nn, load
import numpy as np
from robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
from robustness_experiment_box.verification_module.test_verification_module import TestVerificationModule
from robustness_experiment_box.database.verification_context import VerificationContext

from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.dataset.data_point import DataPoint

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


def test_attack_estimation_module_initialization(attack_estimation_module, attack):
    assert attack_estimation_module.attack == attack

def test_attack_estimation_module_verify_unsat(attack_estimation_module, verification_context):
    epsilon = 0.1
    result = attack_estimation_module.verify(verification_context, epsilon)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == "UNSAT"

def test_attack_estimation_module_verify_sat(attack_estimation_module, verification_context):
    epsilon = 0.5
    result = attack_estimation_module.verify(verification_context, epsilon)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == "SAT"

def test_attack_estimation_module_verify_not_implemented(attack_estimation_module, verification_context):
    class AnotherPropertyGenerator:
        pass
    verification_context.property_generator = AnotherPropertyGenerator()
    epsilon = 0.1
    with pytest.raises(NotImplementedError):
        attack_estimation_module.verify(verification_context, epsilon)