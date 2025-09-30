from unittest.mock import MagicMock

import pytest

from ada_verona.database.epsilon_status import EpsilonStatus
from ada_verona.database.epsilon_value_result import EpsilonValueResult
from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import VerificationResult
from ada_verona.epsilon_value_estimator.iterative_epsilon_value_estimator import (
    IterativeEpsilonValueEstimator,
)


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
    estimator = IterativeEpsilonValueEstimator(verifier = verifier, epsilon_value_list=[0.1, 0.2, 0.3, 0.4, 0.5])
    return estimator

def test_compute_epsilon_value(epsilon_value_estimator, verification_context, verifier):
    # Mock the verifier's verify method to return different results
    verifier.verify.side_effect = [
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
    ]

    result = epsilon_value_estimator.compute_epsilon_value(verification_context, reverse_search=False)
    assert isinstance(result, EpsilonValueResult)
    assert result.epsilon== 0.2
    assert result.smallest_sat_value == 0.3
    assert result.time > 0

def test_iterative_search(epsilon_value_estimator, verification_context, verifier):
    epsilon_status_list = [EpsilonStatus(x, None) for x in epsilon_value_estimator.epsilon_value_list]

    # Mock the verifier's verify method to return different results
    verifier.verify.side_effect = [
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
    ]

    highest_unsat_value, lowest_sat_value, updated_epsilon_status_list = (
        epsilon_value_estimator.iterative_search(verification_context, epsilon_status_list)
    )
    print(highest_unsat_value, lowest_sat_value, updated_epsilon_status_list)
    assert highest_unsat_value == 0.2
    assert lowest_sat_value == 0.3
    assert len(updated_epsilon_status_list) == len(epsilon_value_estimator.epsilon_value_list)
    assert all(isinstance(status, EpsilonStatus) for status in updated_epsilon_status_list)

@pytest.mark.parametrize("reverse_search, expected_highest_unsat, expected_lowest_sat", [
    (True, 0.2, 0.3)    # Descending order
])
def test_compute_epsilon_value_directions(
    epsilon_value_estimator, verification_context, verifier, 
    reverse_search, expected_highest_unsat, expected_lowest_sat
):
    verifier.verify.side_effect = [
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.SAT, took=1.0),
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
        MagicMock(result=VerificationResult.UNSAT, took=1.0),
    ]

    result = epsilon_value_estimator.compute_epsilon_value(
        verification_context, reverse_search=reverse_search
    )
    assert result.epsilon == expected_highest_unsat
    assert result.smallest_sat_value == expected_lowest_sat

