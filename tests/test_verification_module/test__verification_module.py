import pytest

from ada_verona.verification_module.verification_module import VerificationModule


def test_cannot_instantiate_verification_module():
    with pytest.raises(TypeError):
        VerificationModule()

def test_verify_raises_notimplementederror():
    # Call the unbound method so the body executes
    with pytest.raises(NotImplementedError):
        # Pass None for the verification_context to match the signature
        VerificationModule.verify(VerificationModule, None, 0.0)

def test_docstring_is_executed():
    # Accessing __doc__ ensures the docstring line is executed
    assert "Main method to verify an image" in VerificationModule.verify.__doc__