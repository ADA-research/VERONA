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