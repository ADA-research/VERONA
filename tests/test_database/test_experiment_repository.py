import pytest
import pandas as pd
import os
from robustness_experiment_box.database.experiment_repository import ExperimentRepository
from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.database.verification_context import VerificationContext
import torch
from robustness_experiment_box.analysis.report_creator import ReportCreator
import yaml
#TODO: some of these are double fixtures, add them to one file. 

class MockVerificationContext:
    """
    A mock class for VerificationContext to simulate its behavior for testing.
    """

    def get_dict_for_epsilon_result(self):
        return {"mock_key": "mock_value"}
    
    def to_dict(self):
        return {"mock_key": "mock_value"}




@pytest.fixture
def mock_experiment_repository(tmp_path):
    base_path = tmp_path / "experiments"
    base_path.mkdir()
    network_folder = base_path / "networks"
    network_folder.mkdir()
    return ExperimentRepository(base_path=base_path, network_folder=network_folder)

@pytest.fixture
def mock_verification_context():
    return MockVerificationContext()

@pytest.fixture 
def mock_epsilon_value_result(mock_verification_context):
    epsilon = 0.5
    smallest_sat_value = 0.3
    time_taken = 1.23

    result = EpsilonValueResult(
        verification_context=mock_verification_context,
        epsilon=epsilon,
        smallest_sat_value=smallest_sat_value,
        time=time_taken,
    )

    return result




def test_get_act_experiment_path(mock_experiment_repository):
    # Arrange
    mock_experiment_repository.act_experiment_path = mock_experiment_repository.base_path / "test_experiment"

    # Act
    path = mock_experiment_repository.get_act_experiment_path()

    # Assert
    assert path == mock_experiment_repository.base_path / "test_experiment"


def test_get_act_experiment_path_no_experiment(mock_experiment_repository):
    # Act & Assert
    with pytest.raises(Exception, match="No experiment loaded"):
        mock_experiment_repository.get_act_experiment_path()


def test_get_results_path(mock_experiment_repository):
    # Arrange
    mock_experiment_repository.act_experiment_path = mock_experiment_repository.base_path / "test_experiment"

    # Act
    results_path = mock_experiment_repository.get_results_path()

    # Assert
    assert results_path == mock_experiment_repository.base_path / "test_experiment" / "results"


def test_get_tmp_path(mock_experiment_repository):
    # Arrange
    mock_experiment_repository.act_experiment_path = mock_experiment_repository.base_path / "test_experiment"

    # Act
    tmp_path = mock_experiment_repository.get_tmp_path()

    # Assert
    assert tmp_path == mock_experiment_repository.base_path / "test_experiment" / "tmp"


def test_initialize_new_experiment(mock_experiment_repository):
    # Arrange
    experiment_name = "test_experiment"

    # Act
    mock_experiment_repository.initialize_new_experiment(experiment_name)

    # Assert
    assert mock_experiment_repository.act_experiment_path is not None
    assert mock_experiment_repository.act_experiment_path.name.startswith(experiment_name)
    assert (mock_experiment_repository.act_experiment_path / "results").exists()
    assert (mock_experiment_repository.act_experiment_path / "tmp").exists()

    with pytest.raises(Exception, match="Error, there is already a directory with results with the same name, make sure no results will be overwritten"):
        mock_experiment_repository.initialize_new_experiment(experiment_name)


def test_cleanup_tmp_directory(mock_experiment_repository):
 
    experiment_name = "test_experiment"
    mock_experiment_repository.initialize_new_experiment(experiment_name)

    # Create a dummy file in the tmp directory to ensure file.unlink() is covered
    tmp_path = mock_experiment_repository.get_tmp_path()
    tmp_path.mkdir(parents=True, exist_ok=True)
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("temporary content")

    assert tmp_path.exists()
    assert dummy_file.exists()

    # Run the cleanup method
    mock_experiment_repository.cleanup_tmp_directory()

    # Assert the directory and file are both gone
    assert not dummy_file.exists()
    assert not tmp_path.exists()

def test_load_experiment(mock_experiment_repository):
    assert mock_experiment_repository.act_experiment_path is None
    experiment_path = "experiments"
    mock_experiment_repository.load_experiment(experiment_path)
    
    assert mock_experiment_repository.act_experiment_path == mock_experiment_repository.base_path/ experiment_path


def test_get_network_list(mock_experiment_repository):

    experiment_name = "test_experiment"
    mock_experiment_repository.initialize_new_experiment(experiment_name)
    network_path = mock_experiment_repository.network_folder / "network1"
    network_path.mkdir()
    network_list = mock_experiment_repository.get_network_list()

    assert len(network_list) == 1
    assert network_list[0].path == mock_experiment_repository.network_folder /"network1"


def test_save_results(mock_experiment_repository, mock_epsilon_value_result):
    mock_experiment_repository.initialize_new_experiment("test_experiment")
    result_path = mock_experiment_repository.get_results_path() / "result_df.csv"
  
    mock_experiment_repository.save_results([mock_epsilon_value_result,mock_epsilon_value_result])

    assert result_path.exists()
    df = pd.read_csv(result_path, index_col=0)
    assert len(df) == 2
    assert df.iloc[0]["epsilon_value"] == 0.5
    assert df.iloc[0]["smallest_sat_value"] == 0.3
    assert df.iloc[0]["total_time"] == 1.23



def test_save_result(mock_experiment_repository, mock_epsilon_value_result):

    mock_experiment_repository.initialize_new_experiment("test_experiment")
    result_path = mock_experiment_repository.get_results_path() / "result_df.csv"

    mock_experiment_repository.save_result(mock_epsilon_value_result)

    assert result_path.exists()
    df = pd.read_csv(result_path, index_col=0)
    assert len(df) == 1
    assert df.iloc[0]["epsilon_value"] == 0.5
    assert df.iloc[0]["smallest_sat_value"] == 0.3
    assert df.iloc[0]["total_time"] == 1.23
    
    #do it again to test what happes when the result file already exists and make sure it does not override the first results
    mock_experiment_repository.save_result(mock_epsilon_value_result)
    df = pd.read_csv(result_path, index_col=0)
    assert len(df) == 2
    assert df.iloc[1]["epsilon_value"] == 0.5
    assert df.iloc[1]["smallest_sat_value"] == 0.3   




def test_get_file_name(mock_experiment_repository, tmp_path):

    file_path = tmp_path / "example_file.txt"
    file_path.touch()

    file_name = mock_experiment_repository.get_file_name(file_path)

    assert file_name == "example_file"


def test_create_verification_context(mock_experiment_repository, tmp_path):
    mock_experiment_repository.initialize_new_experiment("test_experiment")

    # Create a sample Network, DataPoint, and PropertyGenerator
    network_path = tmp_path / "network.onnx"
    network_path.touch()
    network = Network(network_path)

    data_point = DataPoint(id="1", label=0, data=torch.tensor([1.0, 2.0, 3.0]))

    class DummyPropertyGenerator:
        def generate(self):
            return "dummy_property"

    property_generator = DummyPropertyGenerator()

  
    verification_context = mock_experiment_repository.create_verification_context(
        network, data_point, property_generator
    )

    assert isinstance(verification_context, VerificationContext)
    assert verification_context.network == network
    assert verification_context.data_point == data_point
    assert verification_context.property_generator == property_generator



def test_get_result_df(mock_experiment_repository):

    mock_experiment_repository.initialize_new_experiment("test_experiment")
    result_path = mock_experiment_repository.get_results_path() / "result_df.csv"
    result_data = [
        {"network_path": "help/network_1.onnx", "epsilon_value": 0.5, "result": "SAT", "total_time": 1.23},
        {"network_path": "help/network_2.onnx", "epsilon_value": 0.7, "result": "UNSAT", "total_time": 2.34},
    ]
    results = pd.DataFrame(result_data)
    results.to_csv( result_path, index=True)

    result_df = mock_experiment_repository.get_result_df()
  
    assert not result_df.empty
    assert len(result_df) == 2
    assert "network" in result_df.columns
    assert result_df.iloc[0]["network"] == "network_1"

    # Create a temporary file to simulate the absence of the result file
    temp_file = mock_experiment_repository.get_results_path() / "non_existent_file.csv"
    temp_file.touch()
    # Remove the result file
    os.remove(result_path)
    # Assert that an exception is raised when trying to get the result DataFrame
    with pytest.raises(Exception, match="Error, no result file found at"):
        mock_experiment_repository.get_result_df()


def test_get_per_epsilon_result_df(mock_experiment_repository, tmp_path):

    mock_experiment_repository.initialize_new_experiment("test_experiment")
    tmp_path = mock_experiment_repository.get_tmp_path()
    network_folder = tmp_path / "network_1"
    network_folder.mkdir()
    image_folder = network_folder / "image_1"
    image_folder.mkdir()
    epsilon_data = [
        {"epsilon": 0.5, "result": "SAT", "time": 1.23},
        {"epsilon": 0.7, "result": "UNSAT", "time": 2.34},
    ]
    pd.DataFrame(epsilon_data).to_csv(image_folder / "epsilons_df.csv", index=True)


    per_epsilon_df = mock_experiment_repository.get_per_epsilon_result_df()

    assert not per_epsilon_df.empty
    assert len(per_epsilon_df) == 2
    assert "network" in per_epsilon_df.columns
    assert per_epsilon_df.iloc[0]["network"] == "network_1"
    assert per_epsilon_df.iloc[0]["image"] == "image_1"


def test_save_per_epsilon_result_df(mock_experiment_repository, tmp_path):

    mock_experiment_repository.initialize_new_experiment("test_experiment")
    tmp_path = mock_experiment_repository.get_tmp_path()
    network_folder = tmp_path / "network_1"
    network_folder.mkdir()
    image_folder = network_folder / "image_1"
    image_folder.mkdir()
    epsilon_data = [
        {"epsilon": 0.5, "result": "SAT", "time": 1.23},
        {"epsilon": 0.7, "result": "UNSAT", "time": 2.34},
    ]
    pd.DataFrame(epsilon_data).to_csv(image_folder / "epsilons_df.csv", index=False)

    mock_experiment_repository.save_per_epsilon_result_df()

    per_epsilon_result_path = mock_experiment_repository.get_results_path() / "per_epsilon_results.csv"
    assert per_epsilon_result_path.exists()
    saved_df = pd.read_csv(per_epsilon_result_path)
    assert len(saved_df) == 2
    assert "network" in saved_df.columns
    assert "image" in saved_df.columns


def test_save_verification_context_to_yaml(mock_experiment_repository,mock_verification_context):
    file_path = "verification_context.yaml"
    mock_experiment_repository.save_verification_context_to_yaml(file_path, mock_verification_context)
    assert os.path.exists(file_path)

    with open(file_path) as file:
        data = yaml.safe_load(file)
        assert data["mock_key"] == "mock_value"

    os.remove(file_path)
