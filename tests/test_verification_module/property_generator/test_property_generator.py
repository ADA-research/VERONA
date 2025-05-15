import pytest
import sys
import importlib.util

from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PropertyGenerator()

@pytest.fixture
def dummy_module(tmp_path):
    # Create a temporary module with a dummy subclass
    module_code = '''
class DummySubclass:
    @classmethod
    def from_dict(cls, data):
        instance = cls()
        instance.value = data["value"]
        return instance
    '''
    module_path = tmp_path / "dummy_module.py"
    module_path.write_text(module_code)

    # Add temp directory to sys.path so importlib can find it
    sys.path.insert(0, str(tmp_path))

    yield "dummy_module"

    # Clean up sys.path after test
    sys.path.pop(0)


def test_from_dict_success(dummy_module):
    # Class to test
    class Base:
        @classmethod
        def from_dict(cls, data: dict):
            class_name = data.pop("type", None)
            module_name = data.pop("module", None)

            if not class_name or not module_name:
                raise ValueError("Missing 'class' or 'module' key in dictionary")

            try:
                module = importlib.import_module(module_name)
                subclass = getattr(module, class_name)
            except (ModuleNotFoundError, AttributeError) as e:
                raise ValueError(f"Could not import {class_name} from {module_name}: {e}") from e

            return subclass.from_dict(data)

    input_dict = {
        "type": "DummySubclass",
        "module": dummy_module,
        "value": 42
    }

    obj = Base.from_dict(input_dict.copy())
    assert hasattr(obj, "value")
    assert obj.value == 42

def test_from_dict_missing_keys_raises_value_error():
    class Dummy:
        @classmethod
        def from_dict(cls, data):
            class_name = data.pop("type", None)
            module_name = data.pop("module", None)

            if not class_name or not module_name:
                raise ValueError("Missing 'class' or 'module' key in dictionary")

    with pytest.raises(ValueError, match="Missing 'class' or 'module'"):
        Dummy.from_dict({"value": 42})
        
def test_from_dict_import_failure_raises_value_error():
    class Dummy:
        @classmethod
        def from_dict(cls, data):
            class_name = data.pop("type", None)
            module_name = data.pop("module", None)

            if not class_name or not module_name:
                raise ValueError("Missing 'class' or 'module' key in dictionary")

            try:
                module = importlib.import_module(module_name)
                subclass = getattr(module, class_name)
            except (ModuleNotFoundError, AttributeError) as e:
                raise ValueError(f"Could not import {class_name} from {module_name}: {e}") from e

            return subclass.from_dict(data)

    with pytest.raises(ValueError, match="Could not import NonExistentClass from fake_module"):
        Dummy.from_dict({
            "type": "NonExistentClass",
            "module": "fake_module"
        })