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
    err: str = ""
    stdout: str = ""


CompleteVerificationResult = Result[
    CompleteVerificationData, CompleteVerificationData
]