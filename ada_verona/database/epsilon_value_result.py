from dataclasses import dataclass

from ada_verona.database.verification_context import VerificationContext


@dataclass
class EpsilonValueResult:
    """
    A dataclass defining the verification result of a single verification.
    """

    verification_context: VerificationContext
    epsilon: float
    smallest_sat_value: float
    time: float = None
    verifier: str = None

    def to_dict(self) -> dict:
        """
        Convert the EpsilonValueResult to a dictionary.

        Returns:
            dict: The dictionary representation of the EpsilonValueResult.
        """
        ret = dict(
            **self.verification_context.get_dict_for_epsilon_result(),
            epsilon_value=self.epsilon,
            smallest_sat_value=self.smallest_sat_value,
            total_time=self.time,
            verifier=self.verifier,
        )
        return ret
