from dataclasses import dataclass
from enum import Enum


class VerificationResult(str, Enum):
    """Class for saving the possible verification results.
    At this point we are using the same Result strings for complete verification and attacks.
    """

    UNSAT = "UNSAT"
    SAT = "SAT"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERR"


@dataclass
class ProbabilisticCertificationResult:
    """
    Result class for probabilistic certification methods like randomized smoothing.

    This provides certified robustness guarantees obtained through Monte Carlo sampling
    with Gaussian noise, following the approach of Cohen et al. (2019).

    Note: Unlike traditional verification, probabilistic certification provides
    a certified radius rather than binary SAT/UNSAT/TIMEOUT/ERROR results. The certified radius
    may be 0 if confidence is insufficient (ABSTAIN in Cohen et al.).

    Attributes:
        predicted_class: The class predicted by the smoothed classifier
        certified_radius: The L2 radius within which the prediction is guaranteed to be constant
        confidence: The statistical confidence level (1 - alpha)
        n0: Number of samples used for initial prediction
        n: Number of samples used for certification
        sigma: Noise level used for smoothing
        certification_time: Time taken for the certification process
    """

    predicted_class: int
    certified_radius: float
    confidence: float  # 1 - alpha
    n0: int
    n: int
    sigma: float
    certification_time: float

    def to_dict(self) -> dict:
        """
        Convert the probabilistic certification result to a dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            "predicted_class": self.predicted_class,
            "certified_radius": self.certified_radius,
            "confidence": self.confidence,
            "n0": self.n0,
            "n": self.n,
            "sigma": self.sigma,
            "certification_time": self.certification_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProbabilisticCertificationResult":
        """
        Create a ProbabilisticCertificationResult from a dictionary.

        Args:
            data: Dictionary containing the result data

        Returns:
            ProbabilisticCertificationResult instance
        """
        return cls(
            predicted_class=data["predicted_class"],
            certified_radius=data["certified_radius"],
            confidence=data["confidence"],
            n0=data["n0"],
            n=data["n"],
            sigma=data["sigma"],
            certification_time=data["certification_time"],
        )
