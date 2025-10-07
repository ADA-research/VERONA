from pathlib import Path

import numpy as np
import pytest

from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData
from ada_verona.verification_module.auto_verify_module import (
    parse_counter_example,
    parse_counter_example_label,
)
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.verification_module.property_generator.one2one_property_generator import (
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

@pytest.fixture
def auto_verify_module_fixture(request, auto_verify_module, auto_verify_module_config):
    if request.param == "auto_verify_module":
        return auto_verify_module
    elif request.param == "auto_verify_module_config":
        return auto_verify_module_config

def test_auto_verify_module_initialization(auto_verify_module, verifier):
    assert auto_verify_module.verifier == verifier
    assert auto_verify_module.timeout == 60
    assert auto_verify_module.config is None


@pytest.mark.parametrize(
    "auto_verify_module_fixture", 
    ["auto_verify_module", "auto_verify_module_config"], 
    indirect=True
)
def test_auto_verify_module_verify(auto_verify_module_fixture, verification_context):
    result = auto_verify_module_fixture.verify(verification_context, 0.6)

    assert isinstance(result, CompleteVerificationData)
    assert result.result == "SAT"
    
    result = auto_verify_module_fixture.verify(verification_context, 0.01)

    assert isinstance(result, CompleteVerificationData)
    assert result.result == "UNSAT"
   

def test_parse_counter_example(result, verification_context):
    counter_example = parse_counter_example(result, verification_context)
    assert isinstance(counter_example, np.ndarray)

    assert counter_example.shape == verification_context.data_point.data.shape


def test_parse_counter_example_label(result):
    label = parse_counter_example_label(result)
    assert isinstance(label, int)
    assert label == 0