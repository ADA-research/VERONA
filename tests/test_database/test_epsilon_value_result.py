import pytest
from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult

class MockVerificationContext:
    """
    A mock class for VerificationContext to simulate its behavior for testing.
    """

    def get_dict_for_epsilon_result(self):
        return {"mock_key": "mock_value"}


@pytest.fixture
def mock_verification_context():
    return MockVerificationContext()


def test_epsilon_value_result_initialization(mock_verification_context):
    # Arrange
    epsilon = 0.5
    smallest_sat_value = 0.3
    time_taken = 1.23

    # Act
    result = EpsilonValueResult(
        verification_context=mock_verification_context,
        epsilon=epsilon,
        smallest_sat_value=smallest_sat_value,
        time=time_taken,
    )

    # Assert
    assert result.verification_context == mock_verification_context
    assert result.epsilon == epsilon
    assert result.smallest_sat_value == smallest_sat_value
    assert result.time == time_taken


def test_epsilon_value_result_to_dict(mock_verification_context):
    # Arrange
    epsilon = 0.7
    smallest_sat_value = 0.4
    time_taken = 2.34
    result = EpsilonValueResult(
        verification_context=mock_verification_context,
        epsilon=epsilon,
        smallest_sat_value=smallest_sat_value,
        time=time_taken,
    )

    # Act
    result_dict = result.to_dict()

    # Assert
    assert result_dict == {
        "mock_key": "mock_value",
        "epsilon_value": epsilon,
        "smallest_sat_value": smallest_sat_value,
        "total_time": time_taken,
    }


def test_epsilon_value_result_with_none_time(mock_verification_context):
    # Arrange
    epsilon = 0.9
    smallest_sat_value = 0.6
    time_taken = None

    # Act
    result = EpsilonValueResult(
        verification_context=mock_verification_context,
        epsilon=epsilon,
        smallest_sat_value=smallest_sat_value,
        time=time_taken,
    )

    # Assert
    assert result.time is None
    result_dict = result.to_dict()
    assert result_dict["total_time"] is None