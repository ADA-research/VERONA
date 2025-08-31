import numpy as np
import pytest

from ada_verona.robustness_experiment_box.database.vnnlib_property import VNNLibProperty
from ada_verona.robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)


@pytest.fixture
def property_generator():
    return One2AnyPropertyGenerator(number_classes=10, data_lb=0, data_ub=1)

def test_create_vnnlib_property(property_generator):
    image = np.array([0.5, 0.5, 0.5])
    image_class = 0
    epsilon = 0.1
    vnnlib_property = property_generator.create_vnnlib_property(image, image_class, epsilon)
    
    assert isinstance(vnnlib_property, VNNLibProperty)
    assert vnnlib_property.name == "property_0_0_1"
    assert "; Spec for image and epsilon 0.10000" in vnnlib_property.content

    content = vnnlib_property.content
    assert "; Spec for image and epsilon 0.10000" in content
    assert "(declare-const X_0 Real)" in content
    assert "(declare-const Y_0 Real)" in content
    assert "(assert (or" in content
    assert "(assert (>= X_0 0.40000000))" in content
    assert "(assert (<= X_0 0.60000000))" in content
    assert "(and (>= Y_5 Y_0))" in content

    # Make sure correct number of inputs/outputs declared
    assert content.count("(declare-const X_") == len(image)
    assert content.count("(declare-const Y_") == property_generator.number_classes

def test_get_dict_for_epsilon_result(property_generator):
    result_dict = property_generator.get_dict_for_epsilon_result()
    assert isinstance(result_dict, dict)
    assert result_dict == {}

def test_to_dict(property_generator):
    result_dict = property_generator.to_dict()
    assert isinstance(result_dict, dict)
    print(result_dict)
    assert result_dict == {
        "number_classes": 10,
        "data_lb": 0,
        "data_ub": 1, 
        'type': 'One2AnyPropertyGenerator', 
        'module': ('ada_verona.robustness_experiment_box.verification_module.'
                  'property_generator.one2any_property_generator')
    }

def test_from_dict():
    data = {
        "number_classes": 10,
        "data_lb": 0,
        "data_ub": 1, 
        'type': 'One2AnyPropertyGenerator', 
        'module': ('ada_verona.robustness_experiment_box.verification_module.'
                  'property_generator.one2any_property_generator')
    }
    property_generator = One2AnyPropertyGenerator.from_dict(data)
    assert isinstance(property_generator, One2AnyPropertyGenerator)
    assert property_generator.number_classes == 10
    assert property_generator.data_lb == 0
    assert property_generator.data_ub == 1