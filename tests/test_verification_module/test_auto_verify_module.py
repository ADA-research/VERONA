from pathlib import Path

import numpy as np
import pytest
from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.auto_verify_module import (
    parse_counter_example,
    parse_counter_example_label,
)
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from robustness_experiment_box.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)


@pytest.fixture(params=[One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def property_generator(request):
    return request.param
@pytest.fixture
def tmp_path():
    return Path("/tmp")

@pytest.fixture
def verification_context(network, datapoint, tmp_path, property_generator):
    return VerificationContext(network, datapoint, tmp_path, property_generator)

def test_auto_verify_module_initialization(auto_verify_module, verifier):
    assert auto_verify_module.verifier == verifier
    assert auto_verify_module.timeout == 60
    assert auto_verify_module.config is None

def test_auto_verify_module_verify(auto_verify_module, verification_context):
    epsilon = 0.1
    result = auto_verify_module.verify(verification_context, epsilon)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == "SAT"

def test_parse_counter_example(result, verification_context):
    counter_example = parse_counter_example(result, verification_context)
    assert isinstance(counter_example, np.ndarray)

    assert counter_example.shape == verification_context.data_point.data.shape


def test_parse_counter_example_label(result):
    label = parse_counter_example_label(result)
    assert isinstance(label, int)
    assert label == 0
