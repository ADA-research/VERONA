# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


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
        Network.from_file(dict(name="network", path=Path("test")))

    class ConcreteNetwork(Network):
        def load_pytorch_model(self, *args, **kwargs):
            return super().load_pytorch_model(*args, **kwargs)

        def get_input_shape(self, *args, **kwargs):
            return super().get_input_shape(*args, **kwargs)

        def to_dict(self, *args, **kwargs):
            return super().to_dict(*args, **kwargs)

        @classmethod
        def from_dict(cls, data: dict):
            return super().from_dict(data)

        @property
        def name(self):
            return super().name

        @property
        def path(self):
            return super().path

    inst = ConcreteNetwork()

    with pytest.raises(NotImplementedError):
        _ = inst.name

    with pytest.raises(NotImplementedError):
        _ = inst.path
