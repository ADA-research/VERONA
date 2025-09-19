import json
import os

import pandas as pd
import pytest
import torch
import yaml

from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.database.machine_learning_method.onnx_network import ONNXNetwork
from robustness_experiment_box.database.verification_context import VerificationContext


def test_get_act_experiment_path(experiment_repository):
    experiment_repository.act_experiment_path = experiment_repository.base_path / "test_experiment"

    path = experiment_repository.get_act_experiment_path()

    assert path == experiment_repository.base_path / "test_experiment"


def test_get_act_experiment_path_no_experiment(experiment_repository):
    with pytest.raises(Exception, match="No experiment loaded"):
        experiment_repository.get_act_experiment_path()


def test_get_results_path(experiment_repository):

    experiment_repository.act_experiment_path = experiment_repository.base_path / "test_experiment"

    results_path = experiment_repository.get_results_path()

    assert results_path == experiment_repository.base_path / "test_experiment" / "results"


def test_get_tmp_path(experiment_repository):

    experiment_repository.act_experiment_path = experiment_repository.base_path / "test_experiment"

    tmp_path = experiment_repository.get_tmp_path()

    assert tmp_path == experiment_repository.base_path / "test_experiment" / "tmp"


def test_initialize_new_experiment(experiment_repository):
 
    experiment_name = "test_experiment"

    experiment_repository.initialize_new_experiment(experiment_name)

    assert experiment_repository.act_experiment_path is not None
    assert experiment_repository.act_experiment_path.name.startswith(experiment_name)
    assert (experiment_repository.act_experiment_path / "results").exists()
    assert (experiment_repository.act_experiment_path / "tmp").exists()

    with pytest.raises(Exception, match="Error, there is already a directory with results with the same name,"
                "make sure no results will be overwritten"  ):
        experiment_repository.initialize_new_experiment(experiment_name)


def test_cleanup_tmp_directory(experiment_repository):
 
    experiment_name = "test_experiment"
    experiment_repository.initialize_new_experiment(experiment_name)

    # Create a dummy file in the tmp directory to ensure file.unlink() is covered
    tmp_path = experiment_repository.get_tmp_path()
    tmp_path.mkdir(parents=True, exist_ok=True)
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("temporary content")

    assert tmp_path.exists()
    assert dummy_file.exists()

    # Run the cleanup method
    experiment_repository.cleanup_tmp_directory()

    # Assert the directory and file are both gone
    assert not dummy_file.exists()
    assert not tmp_path.exists()

def test_load_experiment(experiment_repository):
    assert experiment_repository.act_experiment_path is None
    experiment_path = "experiments"
    experiment_repository.load_experiment(experiment_path)
    
    assert experiment_repository.act_experiment_path == experiment_repository.base_path/ experiment_path


def test_get_network_list(experiment_repository):

    experiment_name = "test_experiment"
    experiment_repository.initialize_new_experiment(experiment_name)
    network_path = experiment_repository.network_folder / "network1.onnx"
    network_path.mkdir()
    network_list = experiment_repository.get_network_list()

    assert len(network_list) == 1
    assert network_list[0].path == experiment_repository.network_folder /"network1.onnx"


def test_save_results(experiment_repository, epsilon_value_result):
    experiment_repository.initialize_new_experiment("test_experiment")
    result_path = experiment_repository.get_results_path() / "result_df.csv"
  
    experiment_repository.save_results([epsilon_value_result,epsilon_value_result])

    assert result_path.exists()
    df = pd.read_csv(result_path, index_col=0)
    assert len(df) == 2
    assert df.iloc[0]["epsilon_value"] == 0.5
    assert df.iloc[0]["smallest_sat_value"] == 0.3
    assert df.iloc[0]["total_time"] == 1.23



def test_save_result(experiment_repository, epsilon_value_result):

    experiment_repository.initialize_new_experiment("test_experiment")
    result_path = experiment_repository.get_results_path() / "result_df.csv"

    experiment_repository.save_result(epsilon_value_result)

    assert result_path.exists()
    df = pd.read_csv(result_path, index_col=0)
    assert len(df) == 1
    assert df.iloc[0]["epsilon_value"] == 0.5
    assert df.iloc[0]["smallest_sat_value"] == 0.3
    assert df.iloc[0]["total_time"] == 1.23
    
    #do it again to test what happes when the result 
    # file already exists and make sure it does not override the first results
    experiment_repository.save_result(epsilon_value_result)
    df = pd.read_csv(result_path, index_col=0)
    assert len(df) == 2
    assert df.iloc[1]["epsilon_value"] == 0.5
    assert df.iloc[1]["smallest_sat_value"] == 0.3   




def test_get_file_name(experiment_repository, tmp_path):

    file_path = tmp_path / "example_file.txt"
    file_path.touch()

    file_name = experiment_repository.get_file_name(file_path)

    assert file_name == "example_file"


def test_create_verification_context(experiment_repository, tmp_path):
    experiment_repository.initialize_new_experiment("test_experiment")

    # Create a sample Network, DataPoint, and PropertyGenerator
    network_path = tmp_path / "network.onnx"
    network_path.touch()
    network = ONNXNetwork(network_path)

    data_point = DataPoint(id="1", label=0, data=torch.tensor([1.0, 2.0, 3.0]))

    class DummyPropertyGenerator:
        def generate(self):
            return "dummy_property"

    property_generator = DummyPropertyGenerator()

  
    verification_context = experiment_repository.create_verification_context(
        network, data_point, property_generator
    )

    assert isinstance(verification_context, VerificationContext)
    assert verification_context.network == network
    assert verification_context.data_point == data_point
    assert verification_context.property_generator == property_generator



def test_get_result_df(experiment_repository):

    experiment_repository.initialize_new_experiment("test_experiment")
    result_path = experiment_repository.get_results_path() / "result_df.csv"
    result_data = [
        {"network_path": "help/network_1.onnx", "epsilon_value": 0.5, "result": "SAT", "total_time": 1.23},
        {"network_path": "help/network_2.onnx", "epsilon_value": 0.7, "result": "UNSAT", "total_time": 2.34},
    ]
    results = pd.DataFrame(result_data)
    results.to_csv( result_path, index=True)

    result_df = experiment_repository.get_result_df()
  
    assert not result_df.empty
    assert len(result_df) == 2
    assert "network" in result_df.columns
    assert result_df.iloc[0]["network"] == "network_1"

    # Create a temporary file to simulate the absence of the result file
    temp_file = experiment_repository.get_results_path() / "non_existent_file.csv"
    temp_file.touch()
    # Remove the result file
    os.remove(result_path)
    # Assert that an exception is raised when trying to get the result DataFrame
    with pytest.raises(Exception, match="Error, no result file found at"):
        experiment_repository.get_result_df()


def test_get_per_epsilon_result_df(experiment_repository, tmp_path):

    experiment_repository.initialize_new_experiment("test_experiment")
    tmp_path = experiment_repository.get_tmp_path()
    network_folder = tmp_path / "network_1"
    network_folder.mkdir()
    image_folder = network_folder / "image_1"
    image_folder.mkdir()
    epsilon_data = [
        {"epsilon": 0.5, "result": "SAT", "time": 1.23},
        {"epsilon": 0.7, "result": "UNSAT", "time": 2.34},
    ]
    pd.DataFrame(epsilon_data).to_csv(image_folder / "epsilons_df.csv", index=True)


    per_epsilon_df = experiment_repository.get_per_epsilon_result_df()
    
    assert not per_epsilon_df.empty
    assert len(per_epsilon_df) == 2
    assert "network" in per_epsilon_df.columns
    assert per_epsilon_df.iloc[0]["network"] == "network_1"
    assert per_epsilon_df.iloc[0]["image"] == "image_1"


def test_save_per_epsilon_result_df(experiment_repository, tmp_path):

    experiment_repository.initialize_new_experiment("test_experiment")
    tmp_path = experiment_repository.get_tmp_path()
    network_folder = tmp_path / "network_1"
    network_folder.mkdir()
    image_folder = network_folder / "image_1"
    image_folder.mkdir()
    epsilon_data = [
        {"epsilon": 0.5, "result": "SAT", "time": 1.23},
        {"epsilon": 0.7, "result": "UNSAT", "time": 2.34},
    ]
    pd.DataFrame(epsilon_data).to_csv(image_folder / "epsilons_df.csv", index=False)

    experiment_repository.save_per_epsilon_result_df()

    per_epsilon_result_path = experiment_repository.get_results_path() / "per_epsilon_results.csv"
    assert per_epsilon_result_path.exists()
    saved_df = pd.read_csv(per_epsilon_result_path)
    assert len(saved_df) == 2
    assert "network" in saved_df.columns
    assert "image" in saved_df.columns


def test_save_verification_context_to_yaml(experiment_repository,mock_verification_context):
    file_path = "verification_context.yaml"
    experiment_repository.save_verification_context_to_yaml(file_path, mock_verification_context)
    assert os.path.exists(file_path)

    with open(file_path) as file:
        data = yaml.safe_load(file)
        assert data["mock_key"] == "mock_value"

    os.remove(file_path)





def test_save_configuration_and_read_back(experiment_repository, tmp_path):
    """Ensure JSON configuration is written correctly."""
    experiment_repository.initialize_new_experiment("cfg_experiment")

    cfg = {"a": 1, "b": "two"}
    experiment_repository.save_configuration(cfg)

    cfg_file = experiment_repository.get_act_experiment_path() / "configuration.json"
    assert cfg_file.exists()

    with open(cfg_file) as f:
        data = json.load(f)
    assert data == cfg


def test_save_plots_creates_all_pngs(experiment_repository):
    """
    Cover the save_plots method end-to-end.
    Creates a minimal result_df with required columns so ReportCreator works.
    """
    experiment_repository.initialize_new_experiment("plots_experiment")
    result_path = experiment_repository.get_results_path() / "result_df.csv"

    # minimal but valid dataframe
    df = pd.DataFrame([
        {"network_path": "networks/netA.onnx",
         "epsilon_value": 0.1,
         "result": "SAT",
         "total_time": 0.5},
        {"network_path": "networks/netB.onnx",
         "epsilon_value": 0.3,
         "result": "UNSAT",
         "total_time": 0.3},
    ])
    df.to_csv(result_path)

    experiment_repository.save_plots()

    # All four expected plot files should now exist
    results_dir = experiment_repository.get_results_path()
    for fname in ["hist_figure.png", "boxplot.png", "kde_plot.png", "ecdf_plot.png"]:
        f = results_dir / fname
        assert f.exists() and f.stat().st_size > 0


def test_load_verification_context_from_yaml_roundtrip(tmp_path, experiment_repository, property_generator):

    experiment_repository.initialize_new_experiment("exp")

    # Minimal network file
    net_path = tmp_path / "net.onnx"
    net_path.touch()
    network = ONNXNetwork(net_path)

    # Minimal data point
    data_point = DataPoint(id="1", label=0, data=torch.tensor([1.0]))

    vc_original = VerificationContext(
        network=network,
        data_point=data_point,
        tmp_path=experiment_repository.get_tmp_path(),
        property_generator=property_generator
    )

    yaml_file = tmp_path / "vc.yaml"

   
    experiment_repository.save_verification_context_to_yaml(yaml_file, vc_original)
    vc_loaded = experiment_repository.load_verification_context_from_yaml(yaml_file)

    assert isinstance(vc_loaded, VerificationContext)
    assert vc_loaded.network.path == vc_original.network.path
    assert vc_loaded.data_point.id == vc_original.data_point.id
    assert vc_loaded.data_point.label == vc_original.data_point.label
