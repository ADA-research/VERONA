from enum import Enum

class VerificationResult(str, Enum):
    UNSAT = "UNSAT"
    SAT = "SAT"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERR"
