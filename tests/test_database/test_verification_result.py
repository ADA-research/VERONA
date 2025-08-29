from robustness_experiment_box.database.verification_result import VerificationResult


def test_verification_result_values():
   
    assert VerificationResult.UNSAT == "UNSAT"
    assert VerificationResult.SAT == "SAT"
    assert VerificationResult.TIMEOUT == "TIMEOUT"
    assert VerificationResult.ERROR == "ERR"
    assert VerificationResult.UNKNOWN == "UNKNOWN"


def test_verification_result_membership():
    assert "UNSAT" in VerificationResult.__members__.values()
    assert "SAT" in VerificationResult.__members__.values()
    assert "TIMEOUT" in VerificationResult.__members__.values()
    assert "ERR" in VerificationResult.__members__.values()
    assert "UNKNOWN" in VerificationResult.__members__.values()


def test_verification_result_iteration():
    result_values = [result.value for result in VerificationResult]

    assert result_values == ["UNSAT", "SAT", "TIMEOUT", "ERR", "UNKNOWN"]


def test_verification_result_type():
    assert isinstance(VerificationResult.UNSAT, str)
    assert isinstance(VerificationResult.SAT, str)
    assert isinstance(VerificationResult.TIMEOUT, str)
    assert isinstance(VerificationResult.ERROR, str)
    assert isinstance(VerificationResult.UNKNOWN, str)