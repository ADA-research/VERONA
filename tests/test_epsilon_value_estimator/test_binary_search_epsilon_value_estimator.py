from unittest.mock import MagicMock

import pytest

from ada_verona.database.epsilon_status import EpsilonStatus
from ada_verona.database.epsilon_value_result import EpsilonValueResult
from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import VerificationResult
from ada_verona.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from tests.test_epsilon_value_estimator.conftest import MockVerificationModule


@pytest.fixture
def verification_context():
    context = MagicMock(spec=VerificationContext)
    context.save_result = MagicMock()
    return context

@pytest.fixture
def verifier():
    verifier = MagicMock()
    verifier.verify = MagicMock()
    return verifier

@pytest.fixture
def epsilon_value_estimator(verifier):
    estimator = BinarySearchEpsilonValueEstimator(verifier = verifier, epsilon_value_list=[0.1, 0.2, 0.3, 0.4, 0.5])
    return estimator

class TestBinarySearchEpsilonValueEstimator:

    def test_verifier_gets_called(self, epsilon_value_estimator, verification_context, verifier):
     
        epsilon_value_estimator.compute_epsilon_value(verification_context)

        verifier.verify.assert_called()

    def test_result_class_returned(self, verification_context):
        verifier = MockVerificationModule({0.1 : VerificationResult.SAT})
        estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list=[0.1], verifier=verifier)

        epsilon_value_result = estimator.compute_epsilon_value(verification_context)

        assert isinstance(epsilon_value_result, EpsilonValueResult)

    @pytest.mark.parametrize("epsilon_verification_dict, expected_result", [
    ({0.1 : VerificationResult.ERROR, 0.2 : VerificationResult.ERROR, 0.3: VerificationResult.ERROR}, 0.),
    ({0.1 : VerificationResult.TIMEOUT, 0.2 : VerificationResult.TIMEOUT, 0.3: VerificationResult.TIMEOUT}, 0.),
    ({0.1 : VerificationResult.SAT, 0.2 : VerificationResult.SAT, 0.3: VerificationResult.SAT}, 0),
    ({0.1 : VerificationResult.UNSAT, 0.2 : VerificationResult.SAT, 0.3: VerificationResult.SAT}, 0.1),
    ({0.1 : VerificationResult.UNSAT, 0.2 : VerificationResult.UNSAT, 0.3: VerificationResult.SAT}, 0.2),
    ({0.1 : VerificationResult.ERROR, 0.2 : VerificationResult.ERROR, 0.3: VerificationResult.UNSAT}, 0.3),
    ({0.1 : VerificationResult.UNSAT, 0.2 : VerificationResult.ERROR, 0.3: VerificationResult.ERROR}, 0.1),
    ])
    def test_compute_epsilon_value(self, verification_context, epsilon_verification_dict, expected_result):

        verifier = MockVerificationModule(epsilon_verification_dict)
        estimator = BinarySearchEpsilonValueEstimator(
            epsilon_value_list=list(epsilon_verification_dict.keys()), 
            verifier=verifier
        )

        epsilon_value_result = estimator.compute_epsilon_value(verification_context)

        assert epsilon_value_result.epsilon == expected_result

def test_compute_epsilon_value_correct(epsilon_value_estimator, verification_context, verifier):
    # Mock the verifier's verify method to return different results
    verifier.verify.side_effect = [
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
    ]

    result = epsilon_value_estimator.compute_epsilon_value(verification_context)
    print("this is the result", result)
    assert isinstance(result, EpsilonValueResult)
    assert result.epsilon == 0.2
    assert result.smallest_sat_value == 0.3
    assert result.time > 0

def test_binary_search(epsilon_value_estimator, verification_context, verifier):
    epsilon_status_list = [EpsilonStatus(x, None) for x in epsilon_value_estimator.epsilon_value_list]

    # Mock the verifier's verify method to return different results
    verifier.verify.side_effect = [
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
    ]

    highest_unsat_value, smallest_sat_value = epsilon_value_estimator.binary_search(
        verification_context, epsilon_status_list)

    assert highest_unsat_value == 0.2
    assert smallest_sat_value == 0.3
    assert len(epsilon_status_list) == len(epsilon_value_estimator.epsilon_value_list)
    assert all(isinstance(status, EpsilonStatus) for status in epsilon_status_list)

def test_get_highest_unsat(epsilon_value_estimator):
    epsilon_status_list = [
        EpsilonStatus(0.1, VerificationResult.UNSAT),
        EpsilonStatus(0.2, VerificationResult.UNSAT),
        EpsilonStatus(0.3, VerificationResult.SAT),
        EpsilonStatus(0.4, VerificationResult.SAT),
        EpsilonStatus(0.5, VerificationResult.SAT),
    ]

    highest_unsat_value = epsilon_value_estimator.get_highest_unsat(epsilon_status_list)

    assert highest_unsat_value == 0.2

def test_get_smallest_sat(epsilon_value_estimator):
    epsilon_status_list = [
        EpsilonStatus(0.1, VerificationResult.UNSAT),
        EpsilonStatus(0.2, VerificationResult.UNSAT),
        EpsilonStatus(0.3, VerificationResult.SAT),
        EpsilonStatus(0.4, VerificationResult.SAT),
        EpsilonStatus(0.5, VerificationResult.SAT),
    ]

    smallest_sat_value = epsilon_value_estimator.get_smallest_sat(epsilon_status_list)

    assert smallest_sat_value == 0.3
