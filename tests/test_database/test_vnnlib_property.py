from pathlib import Path

from robustness_experiment_box.database.vnnlib_property import VNNLibProperty


def test_vnnlib_property_initialization():
    # Arrange
    name = "property_1"
    content = "(assert (> x 0))"
    path = Path("/path/to/property.vnnlib")

    # Act
    vnnlib_property = VNNLibProperty(name=name, content=content, path=path)

    # Assert
    assert vnnlib_property.name == name
    assert vnnlib_property.content == content
    assert vnnlib_property.path == path


def test_vnnlib_property_without_path():
    # Arrange
    name = "property_2"
    content = "(assert (< y 1))"

    # Act
    vnnlib_property = VNNLibProperty(name=name, content=content)

    # Assert
    assert vnnlib_property.name == name
    assert vnnlib_property.content == content
    assert vnnlib_property.path is None