import logging
import time

from robustness_experiment_box.database.epsilon_status import EpsilonStatus
from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.verification_result import VerificationResult
from robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator

logger = logging.getLogger(__name__)

class IterativeEpsilonValueEstimator(EpsilonValueEstimator):
    """
    A class to estimate the epsilon value using an iterative search with configurable direction.
    """
    def compute_epsilon_value(
    self,
    verification_context: VerificationContext,
    reverse_search=False,
) -> EpsilonValueResult:
        """
        Compute the epsilon value using an iterative search.

        Args:
            verification_context (VerificationContext): The context for verification.

        Returns:
            EpsilonValueResult: The result of the epsilon value estimation.
        """
        sorted_epsilons = sorted(self.epsilon_value_list, reverse=reverse_search)
        epsilon_status_list = [EpsilonStatus(x, None) for x in sorted_epsilons]
        start_time = time.time()
        highest_unsat_value, lowest_sat_value, epsilon_status_list = self.iterative_search(
            verification_context, epsilon_status_list
        )
        duration = time.time() - start_time
        epsilon_value_result = EpsilonValueResult(verification_context, 
                                                  highest_unsat_value, 
                                                  lowest_sat_value, 
                                                  duration)

        return epsilon_value_result
    
    def iterative_search(self, 
                         verification_context: VerificationContext, 
                        epsilon_status_list: list[EpsilonStatus]) -> tuple[float, float, list]:
        """
        Perform search and determine results based on actual epsilon values. 
        Find the highest UNSAT and smallest SAT epsilon values.

        Args:
            verification_context (VerificationContext): The context for verification.
            epsilon_status_list (list[EpsilonStatus]): The list of epsilon statuses.

        Returns:
            float: The highest UNSAT epsilon value.
            float: The smallest SAT epsilon value.
            list: The epsilon status list.
        """
        
        for status in epsilon_status_list:
            outcome = self.verifier.verify(verification_context, status.value)
            status.result = outcome.result
            status.time = outcome.took
            verification_context.save_result(status)
            logger.info(f"epsilon value: {status.value}, result: {status.result}")

        unsat_values = [x.value for x in epsilon_status_list if x.result == VerificationResult.UNSAT]
        sat_values = [x.value for x in epsilon_status_list if x.result == VerificationResult.SAT]

        highest_unsat = max(unsat_values) if unsat_values else 0
        lowest_sat = min(sat_values) if sat_values else 'undefined'

        
        return highest_unsat, lowest_sat, epsilon_status_list