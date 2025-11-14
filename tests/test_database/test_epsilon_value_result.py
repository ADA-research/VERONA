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

from ada_verona.database.epsilon_value_result import EpsilonValueResult


def test_epsilon_value_result_initialization(mock_verification_context):

    epsilon = 0.5
    smallest_sat_value = 0.3
    time_taken = 1.23


    result = EpsilonValueResult(
        verification_context=mock_verification_context,
        epsilon=epsilon,
        smallest_sat_value=smallest_sat_value,
        time=time_taken,
    )


    assert result.verification_context == mock_verification_context
    assert result.epsilon == epsilon
    assert result.smallest_sat_value == smallest_sat_value
    assert result.time == time_taken


def test_epsilon_value_result_to_dict(mock_verification_context):

    epsilon = 0.7
    smallest_sat_value = 0.4
    time_taken = 2.34
    result = EpsilonValueResult(
        verification_context=mock_verification_context,
        epsilon=epsilon,
        smallest_sat_value=smallest_sat_value,
        time=time_taken,
    )


    result_dict = result.to_dict()


    assert result_dict == {
        "mock_key": "mock_value",
        "epsilon_value": epsilon,
        "smallest_sat_value": smallest_sat_value,
        "total_time": time_taken,
        "verifier": None
    }


