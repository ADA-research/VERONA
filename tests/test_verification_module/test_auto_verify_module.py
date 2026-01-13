# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from result import Err, Ok

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
    "auto_verify_module_fixture", ["auto_verify_module", "auto_verify_module_config"], indirect=True
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


def test_auto_verify_module_verify_sat_with_counter_example(auto_verify_module, verification_context, datapoint):
    """Test that SAT results with counter_example parse the label correctly."""

    formatted_strings = [f"(X_{i} {datapoint.data.flatten()[i]:.4f})" for i in range(28 * 28)]
    counter_example = "\n".join(formatted_strings)
    counter_example += "\n(Y_0 0.1)\n(Y_1 0.9)"

    mock_result = Ok(CompleteVerificationData(result="SAT", counter_example=counter_example, took=10.0))
    auto_verify_module.verifier.verify_property = MagicMock(return_value=mock_result)

    result = auto_verify_module.verify(verification_context, 0.6)

    assert isinstance(result, CompleteVerificationData)
    assert result.result == "SAT"
    assert result.obtained_labels == ["1"]


def test_auto_verify_module_verify_sat_with_counter_example_parse_error(auto_verify_module, verification_context):
    """Test that exception during label parsing is handled gracefully."""

    counter_example = "invalid format that cannot be parsed"

    mock_result = Ok(CompleteVerificationData(result="SAT", counter_example=counter_example, took=10.0))
    auto_verify_module.verifier.verify_property = MagicMock(return_value=mock_result)

    result = auto_verify_module.verify(verification_context, 0.6)

    assert isinstance(result, CompleteVerificationData)
    assert result.result == "SAT"
    assert result.obtained_labels is None


def test_auto_verify_module_verify_error_result(auto_verify_module, verification_context):
    """Test that Err results are handled correctly."""
    error_message = "Verification failed with error"
    mock_result = Err(error_message)
    auto_verify_module.verifier.verify_property = MagicMock(return_value=mock_result)

    result = auto_verify_module.verify(verification_context, 0.6)

    assert result == error_message


def test_auto_verify_module_verify_unsat_sets_obtained_labels_none(auto_verify_module, verification_context):
    """Test that UNSAT results without obtained_labels attribute get it set to None."""
    class MockOutcome:
        def __init__(self):
            self.result = "UNSAT"
            self.took = 10.0
            self.counter_example = None

    outcome = MockOutcome()
    mock_result = Ok(outcome)
    auto_verify_module.verifier.verify_property = MagicMock(return_value=mock_result)

    result = auto_verify_module.verify(verification_context, 0.01)

    assert hasattr(result, "obtained_labels")
    assert result.obtained_labels is None
    assert result.result == "UNSAT"
