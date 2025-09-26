import pytest
import types
import importlib
from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PropertyGenerator()


def test_abstract_methods_raise_notimplementederror():
    # Call the abstract methods on the class itself (unbound)
    with pytest.raises(NotImplementedError):
        PropertyGenerator.create_vnnlib_property(PropertyGenerator,0,0,0)
        
    with pytest.raises(NotImplementedError):
        PropertyGenerator.get_dict_for_epsilon_result(PropertyGenerator)
        
    with pytest.raises(NotImplementedError):
        PropertyGenerator.to_dict(PropertyGenerator)
        



def test_from_dict_success(monkeypatch):
    """It should import the module, get the class and call its from_dict method."""

    called = {}

    # Create a fake module and class
    fake_module = types.ModuleType("fake_module")

    class DummySubclass:
        @classmethod
        def from_dict(cls, data):
            called["data"] = data
            return "ok"

    fake_module.DummySubclass = DummySubclass
    # Insert fake module into sys.modules so importlib can find it
    monkeypatch.setitem(importlib.sys.modules, "fake_module", fake_module)

    data = {
        "type": "DummySubclass",
        "module": "fake_module",
        "foo": 42,
    }

    result = PropertyGenerator.from_dict(data.copy())
    assert result == "ok"
    # Ensure the remaining data (minus type/module) is forwarded
    assert called["data"] == {"foo": 42}


def test_from_dict_missing_keys():
    """If either 'type' or 'module' is missing, ValueError should be raised."""
    with pytest.raises(ValueError, match="Missing 'class' or 'module'"):
        PropertyGenerator.from_dict({"type": "Something"})  # no module
    with pytest.raises(ValueError, match="Missing 'class' or 'module'"):
        PropertyGenerator.from_dict({"module": "fake_module"})  # no type


def test_from_dict_import_errors():
    """It should raise ValueError when module or class cannot be imported."""
    # Non-existent module
    bad_data = {"type": "DoesNotExist", "module": "no_such_module"}
    with pytest.raises(ValueError, match="Could not import"):
        PropertyGenerator.from_dict(bad_data.copy())

    # Module exists but class missing
    import types, sys
    fake_module = types.ModuleType("another_fake")
    sys.modules["another_fake"] = fake_module
    bad_data2 = {"type": "MissingClass", "module": "another_fake"}
    with pytest.raises(ValueError, match="Could not import"):
        PropertyGenerator.from_dict(bad_data2.copy())