from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from result import Err, Ok

from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData
from ada_verona.verification_module.auto_verify_module import (
    AutoVerifyModule,
    parse_counter_example,
    parse_counter_example_label,
)
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)


@pytest.fixture
def mock_verifier():
    with patch("ada_verona.verification_module.auto_verify_module.Verifier") as MockVerifier:
        mock = MockVerifier.return_value
        mock.name = "MockVerifier"
        mock.verify.return_value = CompleteVerificationData("SAT", 2.12, None)
        yield mock

@pytest.fixture
def auto_verify_module(mock_verifier):
    return AutoVerifyModule(mock_verifier, timeout=60)

def test_auto_verify_module_initialization(auto_verify_module, mock_verifier):
    assert auto_verify_module.verifier == mock_verifier
    assert auto_verify_module.timeout == 60
    assert auto_verify_module.config is None


@pytest.fixture(params=[One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def property_generator(request):
    return request.param
@pytest.fixture
def tmp_path():
    return Path("/tmp")

@pytest.fixture
def verification_context(network, datapoint, tmp_path, property_generator):
    return VerificationContext(network, datapoint, tmp_path, property_generator)

def test_auto_verify_module_verify(mock_verifier, verification_context):
    epsilon = 0.1
    result = mock_verifier.verify(verification_context, epsilon)
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
    
    
    
class DummyCounterExample:
    counter_example = """
    (X_0 0.1)
    (X_1 0.2)
    (Y_0 1.0)
    (Y_1 0.0)
    sat
    """

class DummyCounterExampleAlt:
    counter_example = """
    (X_0 0.3)
    (X_1 0.4)
    (Y_0 0.0)
    (Y_1 2.0)
    sat
    """

def test_auto_verify_module_with_config(mock_verifier, verification_context, tmp_path):
    module = AutoVerifyModule(mock_verifier, timeout=10, config=tmp_path / "dummy.cfg")
    mock_verifier.verify_property.return_value = Ok(CompleteVerificationData("UNSAT", 1.0, None))
    result = module.verify(verification_context, 0.2)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == "UNSAT"
    mock_verifier.verify_property.assert_called_once()

def test_auto_verify_module_err_branch(mock_verifier, verification_context):
    module = AutoVerifyModule(mock_verifier, timeout=10)
    mock_verifier.verify_property.return_value = Err("timeout")
    result = module.verify(verification_context, 0.3)
    assert result == "timeout"

def test_parse_counter_example_empty_raises(verification_context):
    result = Ok(type("R", (), {"counter_example": "sat"})())
    with pytest.raises(ValueError):
        parse_counter_example(result, verification_context)

def test_parse_counter_example_label_nonzero():
    result = Ok(DummyCounterExampleAlt())
    label = parse_counter_example_label(result)
    assert label == 1