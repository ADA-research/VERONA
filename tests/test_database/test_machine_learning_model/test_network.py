
from pathlib import Path

import pytest

from ada_verona.database.machine_learning_model.network import Network


def test_cannot_instantiate_network():
    """Ensure Network cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Network()


def test_abstract_methods_raise_not_implemented_error(network):
    with pytest.raises(NotImplementedError):
        Network.load_pytorch_model(Network)
        
    with pytest.raises(NotImplementedError):
        Network.get_input_shape(Network)

    with pytest.raises(NotImplementedError):
        Network.from_dict(dict())

    with pytest.raises(NotImplementedError):
        Network.to_dict(network)

    with pytest.raises(NotImplementedError):
        Network.from_file(dict(name="network",path=Path("test")))