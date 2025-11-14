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

from dataclasses import dataclass

from ada_verona.database.verification_context import VerificationContext


@dataclass
class EpsilonValueResult:
    """
    A dataclass defining the verification result of a single verification.
    """

    verification_context: VerificationContext
    epsilon: float
    smallest_sat_value: float
    time: float = None
    verifier: str = None

    def to_dict(self) -> dict:
        """
        Convert the EpsilonValueResult to a dictionary.

        Returns:
            dict: The dictionary representation of the EpsilonValueResult.
        """
        ret = dict(
            **self.verification_context.get_dict_for_epsilon_result(),
            epsilon_value=self.epsilon,
            smallest_sat_value=self.smallest_sat_value,
            total_time=self.time,
            verifier=self.verifier,
        )
        return ret
