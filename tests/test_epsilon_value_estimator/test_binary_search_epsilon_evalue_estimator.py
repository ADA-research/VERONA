import pytest

from robustness_experiment_box.database.verification_result import VerificationResult
from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult
from robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)

from tests.test_epsilon_value_estimator.conftest import MockVerificationModule


class TestBinarySearchEpsilonValueEstimator:
    def test_verifier_gets_called(self, mocker, verification_context):
        verification_module = MockVerificationModule(None)
        verifier = mocker.Mock(verification_module)
        estimator = BinarySearchEpsilonValueEstimator([0.1], verifier=verifier)

        epsilon_value_result = estimator.compute_epsilon_value(verification_context)

        verifier.verify.assert_called()

    def test_result_class_returned(self, verification_context):
        verifier = MockVerificationModule({0.1: VerificationResult.SAT})
        estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list=[0.1], verifier=verifier)

        epsilon_value_result = estimator.compute_epsilon_value(verification_context)

        assert isinstance(epsilon_value_result, EpsilonValueResult)

    @pytest.mark.parametrize(
        "epsilon_verification_dict, expected_result",
        [
            ({0.1: VerificationResult.ERROR, 0.2: VerificationResult.ERROR, 0.3: VerificationResult.ERROR}, 0.0),
            ({0.1: VerificationResult.TIMEOUT, 0.2: VerificationResult.TIMEOUT, 0.3: VerificationResult.TIMEOUT}, 0.0),
            ({0.1: VerificationResult.SAT, 0.2: VerificationResult.SAT, 0.3: VerificationResult.SAT}, 0),
            ({0.1: VerificationResult.UNSAT, 0.2: VerificationResult.SAT, 0.3: VerificationResult.SAT}, 0.1),
            ({0.1: VerificationResult.UNSAT, 0.2: VerificationResult.UNSAT, 0.3: VerificationResult.SAT}, 0.2),
            ({0.1: VerificationResult.ERROR, 0.2: VerificationResult.ERROR, 0.3: VerificationResult.UNSAT}, 0.3),
            ({0.1: VerificationResult.UNSAT, 0.2: VerificationResult.ERROR, 0.3: VerificationResult.ERROR}, 0.1),
        ],
    )
    def test_compute_epsilon_value(self, verification_context, epsilon_verification_dict, expected_result):
        verifier = MockVerificationModule(epsilon_verification_dict)
        estimator = BinarySearchEpsilonValueEstimator(
            epsilon_value_list=list(epsilon_verification_dict.keys()), verifier=verifier
        )

        epsilon_value_result = estimator.compute_epsilon_value(verification_context)

        assert epsilon_value_result.epsilon == expected_result
