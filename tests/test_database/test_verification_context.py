from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robustness_experiment_box.database.epsilon_status import EpsilonStatus
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from robustness_experiment_box.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)


@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_to_dict(verification_context, tmp_path):
    context_dict = verification_context.to_dict()
    assert isinstance(context_dict, dict)

    assert context_dict['network'] == {'network_path': str(tmp_path / "network.onnx")}
    assert context_dict['data_point']['id'] == "1"
    assert context_dict['data_point']['label'] == 0
    assert np.allclose(context_dict['data_point']["data"],[0.1, 0.2, 0.3], atol = 1e-5)
    assert context_dict['tmp_path'] == str(verification_context.tmp_path)
    assert context_dict['property_generator'] == verification_context.property_generator.to_dict()  
    assert context_dict['save_epsilon_results'] is True


@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_from_dict(tmp_path, verification_context):
    data = {
        'network': {'network_path': tmp_path / "network.onnx"},
        'data_point': {'id': "1", 'label': 0, 'data': [0.1, 0.2, 0.3]}, 
        'tmp_path': str(tmp_path),
        'property_generator': verification_context.property_generator.to_dict(),
        'save_epsilon_results': True
    }

    context = VerificationContext.from_dict(data)
    assert isinstance(context, VerificationContext)
    assert context.network.path == tmp_path / "network.onnx"
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
def test_save_result_per_epsilon(verification_context, experiment_repository):
    
    experiment_repository.initialize_new_experiment("test_experiment")
    
    # First result (triggers the 'else' branch - new file)
    epsilon_status1 = EpsilonStatus(0.1, None)
    verification_context.save_result(epsilon_status1)
    

    # Second result (triggers the 'if result_df_path.exists()' branch - append)
    epsilon_status2 = EpsilonStatus(0.2, None)
    verification_context.save_result(epsilon_status2)

    save_path = Path(verification_context.tmp_path) / "epsilons_df.csv"
    assert save_path.exists()

    df = pd.read_csv(save_path)
    assert len(df) == 2
    assert df.iloc[0]["epsilon_value"] == 0.1
    assert df.iloc[1]["epsilon_value"] == 0.2

@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_delete_tmp_path(verification_context):
   
    tmp_file = verification_context.tmp_path / "tempfile.txt"
    tmp_file.write_text("test content")

    verification_context.tmp_path = tmp_file


    verification_context.delete_tmp_path()

    assert not tmp_file.exists()

@pytest.mark.parametrize("property_generator", [One2AnyPropertyGenerator(), One2OnePropertyGenerator(target_class=0)])
def test_get_dict_for_epsilon_result(verification_context, tmp_path):
    network_file = tmp_path / "network.onnx"
    network_file.touch()  # create the file

    verification_context.tmp_path = Path("/tmp/some_tmp_dir")
    verification_context.tmp_path.mkdir(parents=True, exist_ok=True)

    result_dict = verification_context.get_dict_for_epsilon_result()
    assert result_dict["network_path"] == network_file.resolve()
    assert result_dict["image_id"] == "1"
    assert result_dict["original_label"] == 0
    assert result_dict["tmp_path"] == verification_context.tmp_path

    verification_context.network.path.unlink()
    verification_context.tmp_path.rmdir()