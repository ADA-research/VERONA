import pytest

from ada_verona.verification_module.verification_module import VerificationModule


def test_cannot_instantiate_verification_module():
    with pytest.raises(TypeError):
        VerificationModule()

def test_verify_raises_not_implemented_error():
    with pytest.raises(NotImplementedError):
        VerificationModule.verify(VerificationModule, None, 0.0)

def test_docstring_is_executed():
    assert "Main method to verify an image" in VerificationModule.verify.__doc__