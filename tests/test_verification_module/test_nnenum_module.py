import pytest
import subprocess
from unittest.mock import patch
import torch
from robustness_experiment_box.verification_module.nnenum_module import NnenumModule
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.dataset.data_point import DataPoint

from autoverify.verifier.verification_result import CompleteVerificationData
import numpy as np

from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from robustness_experiment_box.verification_module.property_generator.one2one_property_generator import One2OnePropertyGenerator


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
def nnenum_module():
    return NnenumModule(timeout=60.0)

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