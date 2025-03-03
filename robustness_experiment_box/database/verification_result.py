from enum import Enum


class VerificationResult(str, Enum):
    """ Class for saving the possible verification results.
    At this point we are using the same Result strings for
        complete verification and attacks.
    """
    UNSAT = "UNSAT"
    SAT = "SAT"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERR"
