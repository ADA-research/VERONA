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

import pytest

from ada_verona.database.verification_result import CompleteVerificationData


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
