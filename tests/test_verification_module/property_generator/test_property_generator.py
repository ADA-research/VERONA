import importlib
import sys
import types

import pytest

from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PropertyGenerator()

def test_abstract_methods_raise_notimplementederror():
    with pytest.raises(NotImplementedError):
        PropertyGenerator.create_vnnlib_property(PropertyGenerator, None, None, None)
    with pytest.raises(NotImplementedError):
        PropertyGenerator.get_dict_for_epsilon_result(PropertyGenerator)
    with pytest.raises(NotImplementedError):
        PropertyGenerator.to_dict(PropertyGenerator)



def test_from_dict_success(monkeypatch):
    called = {}

    # Create a fake module and class
    fake_module = types.ModuleType("fake_module")

    class DummySubclass:
        @classmethod
        def from_dict(cls, data):
            called["data"] = data
            return "ok"

    fake_module.DummySubclass = DummySubclass
    monkeypatch.setitem(importlib.sys.modules, "fake_module", fake_module)

    data = {
        "type": "DummySubclass",
        "module": "fake_module",
        "foo": 42,
    }

    result = PropertyGenerator.from_dict(data.copy())
    assert result == "ok"
    assert called["data"] == {"foo": 42}


def test_from_dict_missing_keys():
    with pytest.raises(ValueError, match="Missing 'class' or 'module'"):
        PropertyGenerator.from_dict({"type": "Something"})  
    with pytest.raises(ValueError, match="Missing 'class' or 'module'"):
        PropertyGenerator.from_dict({"module": "fake_module"}) 


def test_from_dict_import_errors():
    bad_data = {"type": "DoesNotExist", "module": "no_such_module"}
    with pytest.raises(ValueError, match="Could not import"):
        PropertyGenerator.from_dict(bad_data.copy())
 
    fake_module = types.ModuleType("another_fake")
    sys.modules["another_fake"] = fake_module
    bad_data2 = {"type": "MissingClass", "module": "another_fake"}
    with pytest.raises(ValueError, match="Could not import"):
        PropertyGenerator.from_dict(bad_data2.copy())


