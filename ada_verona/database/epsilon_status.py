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

import numpy as np

from ada_verona.database.verification_result import CompleteVerificationData, VerificationResult


@dataclass
class EpsilonStatus:
    """
    A class to represent the status of the verification.
    It records the epsilon value, the result (SAT, UNSAT, TIMEOUT, ERROR) and running time.
    """

    value: float
    result: VerificationResult | None
    time: float = None
    verifier: str = None
    obtained_labels: list[int] = None

    def set_values(self, complete_verification_data: CompleteVerificationData):
        """
        Set values from the CompleteVerificationData

        Args:
            complete_verification_data: CompleteVerificationData
        """
        self.result = complete_verification_data.result
        self.time = complete_verification_data.took
        self.obtained_labels = getattr(complete_verification_data, "obtained_labels", None)

    def to_dict(self) -> dict:
        """Convert the EpsilonStatus to a dictionary."""
        obtained_labels_value = None
        if self.obtained_labels is not None:
            if isinstance(self.obtained_labels, np.ndarray):
                obtained_labels_value = self.obtained_labels.flatten().tolist()
            elif isinstance(self.obtained_labels, list):
                obtained_labels_value = self.obtained_labels
            else:
                obtained_labels_value = [self.obtained_labels]

        return dict(
            epsilon_value=self.value,
            result=self.result,
            time=self.time,
            verifier=self.verifier,
            obtained_labels=obtained_labels_value,
        )
