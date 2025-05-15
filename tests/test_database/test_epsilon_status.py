from robustness_experiment_box.database.epsilon_status import EpsilonStatus
from robustness_experiment_box.database.verification_result import VerificationResult


def test_epsilon_status_initialization():
    # Arrange
    epsilon_value = 0.5
    result = VerificationResult.SAT  # Assuming VerificationResult is an Enum or similar
    time_taken = 1.23

    # Act
    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken)

    # Assert
    assert epsilon_status.value == epsilon_value
    assert epsilon_status.result == result
    assert epsilon_status.time == time_taken


def test_epsilon_status_to_dict():
    # Arrange
    epsilon_value = 0.5
    result = VerificationResult.UNSAT  # Assuming VerificationResult is an Enum or similar
    time_taken = 2.34
    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken)

    # Act
    result_dict = epsilon_status.to_dict()

    # Assert
    assert result_dict == {
        "epsilon_value": epsilon_value,
        "result": result,
        "time": time_taken,
    }


def test_epsilon_status_with_none_result():
    # Arrange
    epsilon_value = 0.7
    result = None
    time_taken = None

    # Act
    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken)

    # Assert
    assert epsilon_status.value == epsilon_value
    assert epsilon_status.result is None
    assert epsilon_status.time is None