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


