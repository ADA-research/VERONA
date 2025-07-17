from dataclasses import dataclass
from enum import Enum
from typing import Literal


class VerificationResult(str, Enum):
    """Class for saving the possible verification results.
    At this point we are using the same Result strings for complete verification and attacks.
    """
    UNSAT = "UNSAT"
    SAT = "SAT"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERR"
# TODO: Consider using Enum for VerificationResultString?
VerificationResultString = Literal["SAT", "UNSAT", "TIMEOUT", "ERR", "UNKNOWN"]

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
    counter_example: str
    err: str = ""
    stdout: str = ""

# FIXME: This doesn't make any sense
# It should be something like Result[CompleteVerificationData, ErrorData]
# 
# CompleteVerificationResult = Result[
#     CompleteVerificationData, CompleteVerificationData
# ]
