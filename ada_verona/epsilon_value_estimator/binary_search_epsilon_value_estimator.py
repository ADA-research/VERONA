import logging
import time

from ada_verona.database.epsilon_status import EpsilonStatus
from ada_verona.database.epsilon_value_result import EpsilonValueResult
from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import VerificationResult
from ada_verona.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class BinarySearchEpsilonValueEstimator(EpsilonValueEstimator):
    """
    A class to get the critical epsilon value using binary search.
    """

    def compute_epsilon_value(self, verification_context: VerificationContext) -> EpsilonValueResult:
        """
        Compute the epsilon value using binary search.

        Args:
            verification_context (VerificationContext): The context for verification.

        Returns:
            EpsilonValueResult: The result of the epsilon value estimation.
        """
        epsilon_status_list = [EpsilonStatus(x, None, None, self.verifier.name) for x in self.epsilon_value_list]

        start_time = time.time()
        highest_unsat_value, smallest_sat_value = self.binary_search(verification_context, epsilon_status_list)
        duration = time.time() - start_time
        epsilon_value_result = EpsilonValueResult(
            verification_context=verification_context,
            epsilon=highest_unsat_value,
            smallest_sat_value=smallest_sat_value,
            time=duration,
            verifier = self.verifier.name,
        )

        logger.info(
            f"Verification Context: {verification_context.get_dict_for_epsilon_result()}, "
            f"epsilon_result: {epsilon_value_result.epsilon}"  
        )
        return epsilon_value_result

    def get_highest_unsat(self, epsilon_status_list: list[EpsilonStatus]) -> float:
        """
        Get the highest UNSAT epsilon value from the list.

        Args:
            epsilon_status_list (list[EpsilonStatus]): The list of epsilon statuses.

        Returns:
            float: The highest UNSAT epsilon value.
        """
        highest_unsat = None
        if len([x.result for x in epsilon_status_list if x.result == VerificationResult.UNSAT]) > 0:
            highest_unsat = max(
                [index for index, x in enumerate(epsilon_status_list) if x.result == VerificationResult.UNSAT]
            )

        highest_unsat_value = epsilon_status_list[highest_unsat].value if highest_unsat is not None else 0

        return highest_unsat_value

    def get_smallest_sat(self, epsilon_status_list: list[EpsilonStatus]) -> float:
        """
        Get the smallest SAT epsilon value from the list.

        Args:
            epsilon_status_list (list[EpsilonStatus]): The list of epsilon statuses.

        Returns:
            float: The smallest SAT epsilon value.
        """
        try:
            max_epsilon_value = max([x.value for x in epsilon_status_list])
        except ValueError:
            return 0
        smallest_sat = None

        if len([x.result for x in epsilon_status_list if x.result == VerificationResult.SAT]) > 0:
            smallest_sat = min(
                [index for index, x in enumerate(epsilon_status_list) if x.result == VerificationResult.SAT]
            )

        smallest_sat_value = epsilon_status_list[smallest_sat].value if smallest_sat is not None else max_epsilon_value

        return smallest_sat_value

    def binary_search(
        self, verification_context: VerificationContext, epsilon_status_list: list[EpsilonStatus]
    ) -> float:
        """
        Perform binary search to find the highest UNSAT and smallest SAT epsilon values.

        Args:
            verification_context (VerificationContext): The context for verification.
            epsilon_status_list (list[EpsilonStatus]): The list of epsilon statuses.

        Returns:
            float: The highest UNSAT and smallest SAT epsilon values.
        """
        if len(epsilon_status_list) == 1:
            outcome = self.verifier.verify(verification_context, epsilon_status_list[0].value)
            result = outcome.result
            epsilon_status_list[0].time = outcome.took
            epsilon_status_list[0].result = result
            logger.debug(f"current epsilon value: {epsilon_status_list[0].result}, took: {epsilon_status_list[0].time}")
            verification_context.save_result(epsilon_status_list[0])
            if result == VerificationResult.UNSAT:
                return epsilon_status_list[0].value, self.get_smallest_sat(epsilon_status_list)
            else:
                return 0, self.get_smallest_sat(epsilon_status_list)

        first = 0
        last = len(epsilon_status_list) - 1

        while first <= last:
            midpoint = (first + last) // 2

            if not epsilon_status_list[midpoint].result:
                outcome = self.verifier.verify(verification_context, epsilon_status_list[midpoint].value)
                epsilon_status_list[midpoint].result = outcome.result
                epsilon_status_list[midpoint].time = outcome.took
                verification_context.save_result(epsilon_status_list[midpoint])
                logger.debug(
                    f"current epsilon value: {epsilon_status_list[midpoint].result},"
                    "took: {epsilon_status_list[midpoint].time}"  
                )

            if epsilon_status_list[midpoint].result == VerificationResult.UNSAT:
                first = midpoint + 1
            elif epsilon_status_list[midpoint].result == VerificationResult.SAT:
                last = midpoint - 1
            else:
                epsilon_status_list.pop(midpoint)
                last = last - 1

        logger.debug(f"epsilon status list: {[(x.value, x.result, x.time) for x in epsilon_status_list]}")

        highest_unsat_value = self.get_highest_unsat(epsilon_status_list)
        smallest_sat_value = self.get_smallest_sat(epsilon_status_list)

        return highest_unsat_value, smallest_sat_value
