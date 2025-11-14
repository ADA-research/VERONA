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

from abc import ABC, abstractmethod

from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData


class VerificationModule(ABC):
    @abstractmethod
    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        """Main method to verify an image for a given network and epsilon value"""
        raise NotImplementedError
