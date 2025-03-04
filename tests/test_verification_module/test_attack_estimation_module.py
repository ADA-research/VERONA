import pytest
from autoverify.verifier.verification_result import CompleteVerificationData


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
