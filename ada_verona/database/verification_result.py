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
from enum import Enum
from typing import Literal

from result import Result


class VerificationResult(str, Enum):
    """Class for saving the possible verification results.
    At this point we are using the same Result strings for complete verification and attacks.
    """

    UNSAT = "UNSAT"
    SAT = "SAT"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERR"

"""Classes for data about verification."""


VerificationResultString = Literal["SAT", "UNSAT", "TIMEOUT", "ERR"]


@dataclass
class CompleteVerificationData:
    """Class holding data about a verification run.

    Attributes:
        result: Outcome (e.g. SAT, UNSAT...)
        took: Wallclock time used
        counter_example: Example that violates property (if SAT)
        err: stderr
        stdout: stdout
    """

    result: VerificationResultString
    took: float
    counter_example: str | None = None
    obtained_labels: list[str] = None
    err: str = ""
    stdout: str = ""

CompleteVerificationResult = Result[
    CompleteVerificationData, CompleteVerificationData
]