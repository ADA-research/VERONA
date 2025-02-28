import pytest
import numpy as np
from pathlib import Path
from result import Err, Ok
from torch import load
from robustness_experiment_box.verification_module.auto_verify_module import  parse_counter_example, parse_counter_example_label
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from robustness_experiment_box.verification_module.property_generator.one2one_property_generator import One2OnePropertyGenerator
from autoverify.verifier.verification_result import CompleteVerificationData


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

def test_parse_counter_example(result):
    counter_example = parse_counter_example(result)
    
    assert isinstance(counter_example, np.ndarray)
    assert counter_example.shape == (28, 28)
    assert np.allclose(counter_example.flatten()[:10], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

def test_parse_counter_example_label(result):
    label = parse_counter_example_label(result)
    print(type(label))
    assert isinstance(label, int)
    assert label == 0