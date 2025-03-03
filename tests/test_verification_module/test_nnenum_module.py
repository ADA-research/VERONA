import pytest
import subprocess
from unittest.mock import patch
from torch import load
from result import Err, Ok
from robustness_experiment_box.database.verification_context import VerificationContext


from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from robustness_experiment_box.verification_module.property_generator.one2one_property_generator import One2OnePropertyGenerator



@pytest.fixture
def verification_context(network, datapoint, tmp_path, property_generator):
    return VerificationContext(network=network, data_point=datapoint, tmp_path=tmp_path, property_generator=property_generator)


@pytest.fixture
def result(datapoint):
    data = """
   (X_3 0.1)
   (Y_0 0.2)
    """
    return Ok(CompleteVerificationData(result="SAT", counter_example=data, took =10))


@patch('subprocess.run')
@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_nnenum_module_verify(mock_subprocess_run, nnenum_module, verification_context):
    mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="SAT", stderr="")
    epsilon = 0.1
    result = nnenum_module.verify(verification_context, epsilon)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == "SAT"

@patch('subprocess.run')
@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_nnenum_module_verify_unsat(mock_subprocess_run, nnenum_module, verification_context):
    mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="UNSAT", stderr="")
    epsilon = 0.1
    result = nnenum_module.verify(verification_context, epsilon)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == "UNSAT"

@patch('subprocess.run')
@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_nnenum_module_verify_timeout(mock_subprocess_run, nnenum_module, verification_context):
    mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Timeout")
    epsilon = 0.1
    result = nnenum_module.verify(verification_context, epsilon)
    assert isinstance(result, CompleteVerificationData)
    assert result.result == "TIMEOUT"