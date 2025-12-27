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

import numpy as np

from ada_verona.database.epsilon_status import EpsilonStatus
from ada_verona.database.verification_result import VerificationResult


def test_epsilon_status_initialization():
    epsilon_value = 0.5
    result = VerificationResult.SAT
    time_taken = 1.23

    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken)

    assert epsilon_status.value == epsilon_value
    assert epsilon_status.result == result
    assert epsilon_status.time == time_taken


def test_epsilon_status_to_dict():
    epsilon_value = 0.5
    result = VerificationResult.UNSAT
    time_taken = 2.34
    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken)

    result_dict = epsilon_status.to_dict()

    assert result_dict == {
        "epsilon_value": epsilon_value,
        "result": result,
        "time": time_taken,
        "verifier": None,
        "obtained_labels": None,
    }


def test_set_values(complete_verification_data):
    epsilon_value = 0.5
    result = VerificationResult.UNSAT
    time_taken = 2.34

    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken)
    epsilon_status.set_values(complete_verification_data)

    assert epsilon_status.obtained_labels == complete_verification_data.obtained_labels
    assert epsilon_status.result == complete_verification_data.result
    assert epsilon_status.time == complete_verification_data.took


def test_epsilon_status_to_dict_with_numpy_array():
    """Test to_dict() when obtained_labels is a numpy array."""
    epsilon_value = 0.5
    result = VerificationResult.SAT
    time_taken = 1.25
    obtained_labels = np.array([1, 2, 3])

    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken, obtained_labels=obtained_labels)

    result_dict = epsilon_status.to_dict()

    assert result_dict["obtained_labels"] == [1, 2, 3]
    assert result_dict["epsilon_value"] == epsilon_value
    assert result_dict["result"] == result
    assert result_dict["time"] == time_taken


def test_epsilon_status_to_dict_with_list():
    """Test to_dict() when obtained_labels is a list."""
    epsilon_value = 0.5
    result = VerificationResult.SAT
    time_taken = 1.23
    obtained_labels = [1, 2, 3]

    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken, obtained_labels=obtained_labels)

    result_dict = epsilon_status.to_dict()

    assert result_dict["obtained_labels"] == [1, 2, 3]
    assert result_dict["epsilon_value"] == epsilon_value
    assert result_dict["result"] == result
    assert result_dict["time"] == time_taken


def test_epsilon_status_to_dict_with_string():
    """Test to_dict() when obtained_labels is a string."""
    epsilon_value = 0.5
    result = VerificationResult.SAT
    time_taken = 1.23
    obtained_labels = "1"

    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken, obtained_labels=obtained_labels)

    result_dict = epsilon_status.to_dict()

    assert result_dict["obtained_labels"] == ["1"]
    assert result_dict["epsilon_value"] == epsilon_value
    assert result_dict["result"] == result
    assert result_dict["time"] == time_taken
