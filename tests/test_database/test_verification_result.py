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

from ada_verona.database.verification_result import VerificationResult


def test_verification_result_values():
   
    assert VerificationResult.UNSAT == "UNSAT"
    assert VerificationResult.SAT == "SAT"
    assert VerificationResult.TIMEOUT == "TIMEOUT"
    assert VerificationResult.ERROR == "ERR"


def test_verification_result_membership():
    assert "UNSAT" in VerificationResult.__members__.values()
    assert "SAT" in VerificationResult.__members__.values()
    assert "TIMEOUT" in VerificationResult.__members__.values()
    assert "ERR" in VerificationResult.__members__.values()


def test_verification_result_iteration():
    result_values = [result.value for result in VerificationResult]

    assert result_values == ["UNSAT", "SAT", "TIMEOUT", "ERR"]


def test_verification_result_type():
    assert isinstance(VerificationResult.UNSAT, str)
    assert isinstance(VerificationResult.SAT, str)
    assert isinstance(VerificationResult.TIMEOUT, str)
    assert isinstance(VerificationResult.ERROR, str)