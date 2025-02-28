import pytest
import numpy as np
from pathlib import Path
from result import Err, Ok
import torch
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.test_verification_module import TestVerificationModule
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from robustness_experiment_box.verification_module.property_generator.one2one_property_generator import One2OnePropertyGenerator
from autoverify.verifier.verification_result import CompleteVerificationData



@pytest.fixture
def network():
    return Network("/path/to/network")

@pytest.fixture
def datapoint():
    return DataPoint("1", 0, torch.tensor([0.1, 0.2, 0.3]))  

@pytest.fixture
def verification_context(network, datapoint, tmp_path, property_generator):
    return VerificationContext(network, datapoint, tmp_path, property_generator)

@pytest.fixture
def auto_verify_module(verifier):
    return AutoVerifyModule(verifier, timeout=60)

@pytest.fixture
def verifier():
    return TestVerificationModule()

def test_auto_verify_module_initialization(auto_verify_module, verifier):
    assert auto_verify_module.verifier == verifier
    assert auto_verify_module.timeout == 60
    assert auto_verify_module.config is None

@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_auto_verify_module_verify(auto_verify_module, verification_context):
    epsilon = 0.1
    result = auto_verify_module.verify(verification_context, epsilon)

    assert isinstance(result, CompleteVerificationData)
    assert result.result == "SAT"

def test_parse_counter_example(auto_verify_module):
    result = Ok(CompleteVerificationData(result="SAT", counter_example=np.array([0.1, 0.2, 0.3]), took = 10))
    counter_example = auto_verify_module.parse_counter_example(result)
    
    assert isinstance(counter_example, np.ndarray)
    assert counter_example.shape == (28, 28)

def test_parse_counter_example_label():
    result = Ok(CompleteVerificationData(result="SAT", counter_example=np.array([0.1, 0.2, 0.3]), took =10))
    label = auto_verify_module.parse_counter_example_label(result)

    assert isinstance(label, int)
    assert label == 1