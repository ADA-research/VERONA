from dataclasses import dataclass

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
        self.obtained_labels = complete_verification_data.obtained_labels

    def to_dict(self) -> dict:
        """
        Convert the EpsilonStatus to a dictionary.

        Returns:
            dict: The dictionary representation of the EpsilonStatus.
        """
        return dict(epsilon_value=self.value, result=self.result, time=self.time, verifier=self.verifier, 
                    obtained_labels=self.obtained_labels.flatten().tolist() if self.obtained_labels else None)
