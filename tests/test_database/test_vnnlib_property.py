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

from ada_verona.database.vnnlib_property import VNNLibProperty


def test_vnnlib_property_initialization():
    name = "property_1"
    content = "(assert (> x 0))"
    path = Path("/path/to/property.vnnlib")

    vnnlib_property = VNNLibProperty(name=name, content=content, path=path)

    assert vnnlib_property.name == name
    assert vnnlib_property.content == content
    assert vnnlib_property.path == path


def test_vnnlib_property_without_path():
    name = "property_2"
    content = "(assert (< y 1))"

    vnnlib_property = VNNLibProperty(name=name, content=content)

    assert vnnlib_property.name == name
    assert vnnlib_property.content == content
    assert vnnlib_property.path is None