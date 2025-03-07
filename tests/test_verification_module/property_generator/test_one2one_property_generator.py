import numpy as np
import pytest

from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
from robustness_experiment_box.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)


@pytest.fixture
def property_generator():
    return One2OnePropertyGenerator(target_class=1, number_classes=10, data_lb=0, data_ub=1)

def test_create_vnnlib_property(property_generator):
    image = np.array([0.5, 0.5, 0.5])
    image_class = 0
    epsilon = 0.1
    vnnlib_property = property_generator.create_vnnlib_property(image, image_class, epsilon)
    assert isinstance(vnnlib_property, VNNLibProperty)
    assert vnnlib_property.name == "property_0_0_1"
    assert "; Spec for image and epsilon 0.10000" in vnnlib_property.content

def test_get_dict_for_epsilon_result(property_generator):
    result_dict = property_generator.get_dict_for_epsilon_result()
    assert isinstance(result_dict, dict)
    assert result_dict == {"target_class": 1}

def test_to_dict(property_generator):
    result_dict = property_generator.to_dict()
    assert isinstance(result_dict, dict)
    assert result_dict == {
        "target_class": 1,
        "number_classes": 10,
        "data_lb": 0,
        "data_ub": 1, 
        'type': 'One2OnePropertyGenerator', 
        'module': 'robustness_experiment_box.verification_module.property_generator.one2one_property_generator'
    }

def test_from_dict():
    data = {
        "target_class": 1,
        "number_classes": 10,
        "data_lb": 0,
        "data_ub": 1, 
        'type': 'One2OnePropertyGenerator', 
        'module': 'robustness_experiment_box.verification_module.property_generator.one2one_property_generator'
    }
    property_generator = One2OnePropertyGenerator.from_dict(data)
    assert isinstance(property_generator, One2OnePropertyGenerator)
    assert property_generator.target_class == 1
    assert property_generator.number_classes == 10
    assert property_generator.data_lb == 0
    assert property_generator.data_ub == 1