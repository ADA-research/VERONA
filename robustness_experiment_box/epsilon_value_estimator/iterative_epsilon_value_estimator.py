import logging
import time
from robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator  # noqa: E501
from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult  # noqa: E501
from robustness_experiment_box.database.verification_context import VerificationContext  # noqa: E501
from robustness_experiment_box.database.verification_result import VerificationResult  # noqa: E501
from robustness_experiment_box.database.epsilon_status import EpsilonStatus

logger = logging.getLogger(__name__)


class IterativeEpsilonValueEstimator(EpsilonValueEstimator):
    """
    A class to estimate the epsilon value using an iterative search.
    """

    def compute_epsilon_value(self,
                              verification_context: VerificationContext
                              ) -> EpsilonValueResult:
        """
        Compute the epsilon value using an iterative search.

        Args:
            verification_context (VerificationContext):
                    The context for verification.

        Returns:
            EpsilonValueResult: The result of the epsilon value estimation.
        """
        epsilon_status_list = [
            EpsilonStatus(x, None) for x in self.epsilon_value_list
        ]
        start_time = time.time()
        highest_unsat_value, lowest_sat_value, epsilon_status_list =\
            self.iterative_search(
                verification_context, epsilon_status_list
            )
        duration = time.time() - start_time
        epsilon_value_result = EpsilonValueResult(
            verification_context, highest_unsat_value,
            lowest_sat_value, duration
        )

        return epsilon_value_result

    def iterative_search(self,
                         verification_context: VerificationContext,
                         epsilon_status_list: list[EpsilonStatus]
                         ) -> float:
        """
        Perform an iterative search to find the highest UNSAT
        and smallest SAT epsilon values.

        Args:
            verification_context (VerificationContext):
                The context for verification.
            epsilon_status_list (list[EpsilonStatus]):
                The list of epsilon statuses.

        Returns:
            float: The highest UNSAT and smallest SAT epsilon values
            and the status list.
        """
        for index in range(0, len(epsilon_status_list)):

            outcome = self.verifier.verify(
                verification_context,
                epsilon_status_list[index].value)
            result = outcome.result
            epsilon_status_list[index].result = result
            epsilon_status_list[index].time = outcome.took
            verification_context.save_result(epsilon_status_list[index])
            logger.info(f"epsilon value: {epsilon_status_list[index].value}, "
                        f"result: {result}")

        highest_unsat = None

        if len([x for x in epsilon_status_list
                if x.result == VerificationResult.UNSAT]) > 0:
            highest_unsat = max([index for index,
                                x in enumerate(epsilon_status_list)
                                if x.result == VerificationResult.UNSAT])

        highest_unsat_value = (epsilon_status_list[highest_unsat].value
                               if highest_unsat is not None else 0)

        lowest_sat = None

        if len([x for x in epsilon_status_list
                if x.result == VerificationResult.SAT]) > 0:
            lowest_sat = min([index for index,
                              x in enumerate(epsilon_status_list)
                              if x.result == VerificationResult.SAT])

        lowest_sat_value = (epsilon_status_list[lowest_sat].value
                            if lowest_sat is not None else "undefined")

        return highest_unsat_value, lowest_sat_value, epsilon_status_list
