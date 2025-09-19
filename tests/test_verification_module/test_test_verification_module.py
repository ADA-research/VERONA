import pytest
import torch
from autoverify.verifier.verification_result import CompleteVerificationData
from result import Ok
from torch import tensor

from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.database.machine_learning_method.onnx_network import ONNXNetwork
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.verification_result import VerificationResult
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from robustness_experiment_box.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)
from robustness_experiment_box.verification_module.test_verification_module import TestVerificationModule
from robustness_experiment_box.verification_module.verification_module import VerificationModule


def test_cannot_instantiate_verification_module():
    """Ensure VerificationModule cannot be instantiated directly."""
    with pytest.raises(TypeError):
        VerificationModule()



@pytest.fixture
def network(tmp_path):
    path = tmp_path / "network.onnx"
    path.touch()
    return ONNXNetwork(path)

@pytest.fixture
def datapoint(tmp_path):
    return DataPoint("1", 0, torch.tensor([0.1, 0.2, 0.3]))

@pytest.fixture(params=[
    One2AnyPropertyGenerator(),
    One2OnePropertyGenerator(target_class=0)
])

def property_generator(request):
    return request.param

@pytest.fixture
def verification_context(network, datapoint, tmp_path, property_generator):
    return VerificationContext(network, datapoint, tmp_path, property_generator)

@pytest.fixture
def test_verification_module():
    return TestVerificationModule()

def test_verify_sat(verification_context, test_verification_module):
    result = test_verification_module.verify(verification_context, epsilon=0.6)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == VerificationResult.SAT
    assert result.took == 10.0

def test_verify_unsat(verification_context, test_verification_module):
    result = test_verification_module.verify(verification_context, epsilon=0.4)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == VerificationResult.UNSAT
    assert result.took == 10.0

def test_verify_network_path_not_found(test_verification_module, datapoint, tmp_path, property_generator):
    invalid_context = VerificationContext(
        type("MockNetwork", (), {"path": tmp_path / "nonexistent.onnx"}),
         datapoint, tmp_path, property_generator
        
    )
    with pytest.raises(Exception, match=r"network path not found"):
        test_verification_module.verify(invalid_context, epsilon=0.6)



def test_verify_property(test_verification_module, tmp_path):
    network_path = tmp_path / "network.onnx"
    vnnlib_path = tmp_path / "property.vnnlib"
    network_path.touch()
    vnnlib_path.touch()

    result = test_verification_module.verify_property(network_path, vnnlib_path, timeout=10)
    assert isinstance(result, Ok)
    assert result.value.result == VerificationResult.SAT
    assert result.value.took == 10.0

def test_verify_property_network_path_not_found(test_verification_module, tmp_path):
    vnnlib_path = tmp_path / "property.vnnlib"
    vnnlib_path.touch()

    with pytest.raises(Exception, match=r"network path not found"):
        test_verification_module.verify_property(tmp_path / "nonexistent.onnx", vnnlib_path, timeout=10)

def test_execute_with_high_epsilon(test_verification_module):
    data_on_device = tensor([1.0, 2.0, 3.0])
    target_on_device = tensor([0])
    epsilon = 0.6

    result = test_verification_module.execute(None, data_on_device, target_on_device, epsilon)
    assert not torch.equal(result, data_on_device)

def test_execute_with_low_epsilon(test_verification_module):
    data_on_device = tensor([1.0, 2.0, 3.0])
    target_on_device = tensor([0])
    epsilon = 0.4

    result = test_verification_module.execute(None, data_on_device, target_on_device, epsilon)
    assert torch.equal(result, data_on_device)