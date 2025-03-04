from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.database.epsilon_status import EpsilonStatus
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from robustness_experiment_box.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)


@pytest.fixture
def network():
    return Network("/path/to/network")


@pytest.fixture
def datapoint():
    return DataPoint("1", 0, torch.tensor([0.1, 0.2, 0.3]))  


@pytest.fixture
def verification_context(network, datapoint, tmp_path, property_generator):
    return VerificationContext(network, datapoint, tmp_path, property_generator)


@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_to_dict(verification_context):
    context_dict = verification_context.to_dict()
    assert isinstance(context_dict, dict)
    assert context_dict['network'] == {'network_path': '/path/to/network'}
    assert context_dict['data_point']['id'] == "1"
    assert context_dict['data_point']['label'] == 0
    assert np.allclose(context_dict['data_point']["data"],[0.1, 0.2, 0.3], atol = 1e-5)
    assert context_dict['tmp_path'] == str(verification_context.tmp_path)
    assert context_dict['property_generator'] == verification_context.property_generator.to_dict()  
    assert context_dict['save_epsilon_results'] is True


@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_from_dict(tmp_path, verification_context):
    data = {
        'network': {'network_path': '/path/to/network'},
        'data_point': {'id': "1", 'label': 0, 'data': [0.1, 0.2, 0.3]}, 
        'tmp_path': str(tmp_path),
        'property_generator': verification_context.property_generator.to_dict(),
        'save_epsilon_results': True
    }

    context = VerificationContext.from_dict(data)
    assert isinstance(context, VerificationContext)
    assert context.network.path == Path('/path/to/network')
    assert context.data_point.id == "1"
    assert context.data_point.label == 0
    assert np.allclose(context.data_point.data.numpy(), [0.1, 0.2, 0.3], atol=1e-5)  
    assert str(context.tmp_path) == str(tmp_path)
    assert context.save_epsilon_results is True


@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_save_vnnlib_property(verification_context):
    vnnlib_property = VNNLibProperty(name="test_property", content="test_content")
    verification_context.save_vnnlib_property(vnnlib_property)
    save_path = Path(verification_context.tmp_path) / "test_property.vnnlib"
    assert save_path.exists()
    with open(save_path) as f:
        content = f.read()
    assert content == "test_content"


@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_save_status_list(verification_context):
    epsilon_status_list = [EpsilonStatus(0.1, None), EpsilonStatus(0.2, None)]
    verification_context.save_status_list(epsilon_status_list)
    save_path = Path(verification_context.tmp_path) / "epsilon_results.csv"
    assert save_path.exists()
    df = pd.read_csv(save_path)
    assert len(df) == 2


@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_save_result(verification_context):
    epsilon_status = EpsilonStatus(0.1, None)
    verification_context.save_result(epsilon_status)
    save_path = Path(verification_context.tmp_path) / "epsilons_df.csv"
    assert save_path.exists()
    df = pd.read_csv(save_path)
    assert len(df) == 1
