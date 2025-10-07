from pathlib import Path

import pytest

from ada_verona.database.dataset.image_file_dataset import ImageFileDataset
from ada_verona.database.machine_learning_model.onnx_network import ONNXNetwork
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler


@pytest.fixture
def network():
    return ONNXNetwork("./example_experiment/data/networks/mnist-net_256x2.onnx")

def test_init():
    sampler = PredictionsBasedSampler()
    assert sampler.sample_correct_predictions is True
    sampler = PredictionsBasedSampler(sample_correct_predictions=True)
    assert sampler.sample_correct_predictions is True
    sampler = PredictionsBasedSampler(sample_correct_predictions=False)
    assert sampler.sample_correct_predictions is False
    
@pytest.fixture
def dataset():
    dataset = ImageFileDataset(image_folder=Path("./example_experiment/data/images"),
                                label_file=Path("./example_experiment/data/image_labels.csv"))
    return dataset


def test_sample_correct_predictions( network, dataset):
    sampler = PredictionsBasedSampler(sample_correct_predictions=True)

    sampled_dataset = sampler.sample(network, dataset)
    selected_indices = [data_point.id for data_point in sampled_dataset]

    assert selected_indices == [ x.id for x in dataset._id_indices if x.id != 'mnist_train_80']


def test_sample_incorrect_predictions(network, dataset):
    sampler = PredictionsBasedSampler(sample_correct_predictions=False)

    sampled_dataset = sampler.sample(network, dataset)
    selected_indices = [data_point.id for data_point in sampled_dataset]

    assert selected_indices == ['mnist_train_80']

def test_sample_network_prediction_failure(dataset, network):

    sampler = PredictionsBasedSampler(sample_correct_predictions=True)

    # Temporarily override get_input_shape to return a shape that will fail reshaping
    network.get_input_shape = lambda: (100, 100)

    with pytest.raises(Exception, match=r"Creating prediction for network .* failed"):
        sampler.sample(network, dataset)