import pytest
import torch
from torch import nn
from robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
from robustness_experiment_box.verification_module.attacks.attack import Attack
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.verification_result import VerificationResult
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from autoverify.verifier.verification_result import CompleteVerificationData

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

class SimpleAttack(Attack):
    def execute(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
        return data + epsilon * torch.sign(data.grad)

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def data():
    return torch.randn(1, 10)

@pytest.fixture
def target():
    return torch.tensor([1])

@pytest.fixture
def attack():
    return SimpleAttack()

@pytest.fixture
def verification_context(model, data, target):
    class Network:
        def load_pytorch_model(self):
            return model

    class DataPoint:
        def __init__(self, label, data):
            self.label = label
            self.data = data

    network = Network()
    data_point = DataPoint(target.item(), data)
    tmp_path = None
    property_generator = One2AnyPropertyGenerator()
    return VerificationContext(network, data_point, tmp_path, property_generator)

@pytest.fixture
def attack_estimation_module(attack):
    return AttackEstimationModule(attack)

def test_attack_estimation_module_initialization(attack_estimation_module, attack):
    assert attack_estimation_module.attack == attack

def test_attack_estimation_module_verify_unsat(attack_estimation_module, verification_context):
    epsilon = 0.1
    verification_context.data_point.label = 1
    verification_context.data_point.data = torch.randn(1, 10)
    result = attack_estimation_module.verify(verification_context, epsilon)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == VerificationResult.UNSAT

def test_attack_estimation_module_verify_sat(attack_estimation_module, verification_context):
    epsilon = 0.1
    verification_context.data_point.label = 0
    verification_context.data_point.data = torch.randn(1, 10)
    result = attack_estimation_module.verify(verification_context, epsilon)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == VerificationResult.SAT

def test_attack_estimation_module_verify_not_implemented(attack_estimation_module, verification_context):
    class AnotherPropertyGenerator:
        pass

    verification_context.property_generator = AnotherPropertyGenerator()
    epsilon = 0.1
    with pytest.raises(NotImplementedError):
        attack_estimation_module.verify(verification_context, epsilon)