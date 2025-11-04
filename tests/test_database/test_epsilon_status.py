from ada_verona.database.epsilon_status import EpsilonStatus
from ada_verona.database.verification_result import VerificationResult


def test_epsilon_status_initialization():

    epsilon_value = 0.5
    result = VerificationResult.SAT 
    time_taken = 1.23

    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken)

    assert epsilon_status.value == epsilon_value
    assert epsilon_status.result == result
    assert epsilon_status.time == time_taken


def test_epsilon_status_to_dict():
    epsilon_value = 0.5
    result = VerificationResult.UNSAT 
    time_taken = 2.34
    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken)


    result_dict = epsilon_status.to_dict()


    assert result_dict == {
        "epsilon_value": epsilon_value,
        "result": result,
        "time": time_taken,
        "verifier": None,
        "obtained_labels": None,
    }

def test_set_values(complete_verification_data):
    epsilon_value = 0.5
    result = VerificationResult.UNSAT 
    time_taken = 2.34

    epsilon_status = EpsilonStatus(value=epsilon_value, result=result, time=time_taken)
    epsilon_status.set_values(complete_verification_data)

    assert epsilon_status.obtained_labels == complete_verification_data.obtained_labels
    assert epsilon_status.result == complete_verification_data.result
    assert epsilon_status.time == complete_verification_data.took

