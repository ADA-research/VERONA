
import importlib

import pytest
import torch
from torch import Tensor

from ada_verona.verification_module.attacks.auto_attack_wrapper import AutoAttackWrapper


class FakeAutoAttack:
    """
    Minimal fake of the real AutoAttack used for testing.
    Records init args and returns a predictable tensor from run_standard_evaluation.
    """

    def __init__(self, model, norm, eps, version, device, verbose):
        # Save for assertions
        self.model = model
        self.norm = norm
        self.eps = eps
        self.version = version
        self.device = device
        self.verbose = verbose
        self.run_called_with = None

    def run_standard_evaluation(self, data: Tensor, target: Tensor) -> Tensor:
        # record what we got and return a perturbed tensor
        # ensure the returned tensor shape matches input batch
        self.run_called_with = (data, target)
        # create a simple deterministic "perturbation": add eps to all values (eps may be float)
        # data might be float tensor -- ensure dtype float
        eps_val = float(self.eps)
        return (data + eps_val).to(data.dtype)


def test_init_sets_attributes_and_name():
    wrapper = AutoAttackWrapper(device="cpu", norm="L2", version="testver", verbose=True)

    assert wrapper.device == "cpu"
    assert wrapper.norm == "L2"
    assert wrapper.version == "testver"
    assert wrapper.verbose is True
    assert "AutoAttackWrapper" in wrapper.name
    assert "norm=L2" in wrapper.name
    assert "version=testver" in wrapper.name


def test_execute_calls_autoattack_and_returns_tensor(monkeypatch):
    """
    Monkeypatch the AutoAttack used inside the wrapper module to our FakeAutoAttack.
    Verify that execute:
      - constructs the AutoAttack with the wrapper's params,
      - calls run_standard_evaluation with the unsqueezed input,
      - returns a tensor moved to the wrapper device.
    """

    # Import the module object that defines AutoAttackWrapper so we can patch its AutoAttack name.
    aa_module = importlib.import_module(
        "ada_verona.verification_module.attacks.auto_attack_wrapper"
    )

    # Patch the AutoAttack symbol in that module to our fake implementation
    monkeypatch.setattr(aa_module, "AutoAttack", FakeAutoAttack)

    # Create a simple dummy model (we don't call it, but passed for signature)
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = DummyModel()

    # Use cpu device for tests to avoid CUDA requirement
    wrapper = AutoAttackWrapper(device="cpu", norm="Linf", version="standard", verbose=False)

    # Create a data tensor with shape (C, H, W) so wrapper will unsqueeze to (1, C, H, W)
    data = torch.zeros((3, 8, 8), dtype=torch.float32)
    # Target should be a 1D tensor with batch size after unsqueeze -> batch size 1
    target = torch.tensor([7], dtype=torch.long)

    # Call execute
    out = wrapper.execute(model=model, data=data, target=target, epsilon=0.05)

    # The output must be a torch.Tensor
    assert isinstance(out, torch.Tensor)

    # The output should be on CPU (the wrapper calls .to(self.device))
    assert out.device.type == "cpu"

    # Because FakeAutoAttack adds eps to data, and wrapper unsqueezes before calling,
    # the returned tensor has a batch dimension. We expect shape (1, C, H, W)
    assert out.dim() == 4
    assert out.shape[0] == 1
    assert out.shape[1:] == data.shape

    # The values should have eps added
    assert torch.allclose(out[0], data + 0.05, atol=1e-6)

    # Also assert that FakeAutoAttack received the right constructor args
    # We can locate the FakeAutoAttack instance by creating one via constructing another wrapper
    # But simpler: re-create and inspect by calling execute again and reading the object used:
    # Our FakeAutoAttack instance is created inside wrapper.execute; to inspect it, patch FakeAutoAttack
    # to store last instance as a module-level variable. We'll do that here by monkeypatching differently.

def test_execute_records_call_arguments(monkeypatch):
    """
    Ensure the AutoAttack constructor receives the wrapper's parameters
    and that run_standard_evaluation is called with the unsqueezed data and provided target.
    """

    # We'll make a Fake that stores the instance globally so we can assert on it.
    created = {}

    class RecordingFakeAutoAttack(FakeAutoAttack):
        def __init__(self, model, norm, eps, version, device, verbose):
            super().__init__(model, norm, eps, version, device, verbose)
            created["instance"] = self

    aa_module = importlib.import_module(
        "ada_verona.verification_module.attacks.auto_attack_wrapper"
    )
    monkeypatch.setattr(aa_module, "AutoAttack", RecordingFakeAutoAttack)

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = DummyModel()
    wrapper = AutoAttackWrapper(device="cpu", norm="Linf", version="standard", verbose=True)

    data = torch.ones((1, 2, 2), dtype=torch.float32)  # (C,H,W)
    target = torch.tensor([3], dtype=torch.long)

    wrapper.execute(model=model, data=data, target=target, epsilon=0.2)

    # Ensure an instance was created
    assert "instance" in created
    inst = created["instance"]

    # constructor args
    assert inst.model is model
    assert inst.norm == "Linf"
    assert pytest.approx(inst.eps, rel=1e-6) == 0.2
    assert inst.version == "standard"
    assert inst.device == "cpu"
    assert inst.verbose is True

    # run_standard_evaluation was called with data unsqueezed to shape (1, C, H, W)
    called_data, called_target = inst.run_called_with
    assert isinstance(called_data, torch.Tensor)
    assert called_data.shape[0] == 1  # batch dim present
    # the called_target should equal the provided target
    assert torch.equal(called_target, target)
