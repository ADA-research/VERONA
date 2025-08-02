from dataclasses import dataclass
from enum import Enum
from typing import Literal

from result import Result


class VerificationResult(str, Enum):
    UNSAT = "UNSAT"
    SAT = "SAT"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERR"
    UNKNOWN = "UNKNOWN"

"""Classes for data about verification."""
# TODO: Enum?
VerificationResultString = Literal["SAT", "UNSAT", "TIMEOUT", "ERR"]


@dataclass
class CompleteVerificationData:
    """Class holding data about a verification run.

    Attributes:
        result: Outcome (e.g. SAT, UNSAT...)
        took: Walltime spent
        counter_example: Example that violates property (if SAT)
        err: stderr
        stdout: stdout
    """

    result: VerificationResultString
    took: float
    counter_example: str | None = None
    err: str = ""
    stdout: str = ""

# FIXME: This doesnt make any sense
# It should be something like Result[C.V.D., ErrorData]
CompleteVerificationResult = Result[
    CompleteVerificationData, CompleteVerificationData
]

