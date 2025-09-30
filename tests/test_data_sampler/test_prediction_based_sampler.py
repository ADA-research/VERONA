from pathlib import Path

import pytest

from ada_verona.database.dataset.image_file_dataset import ImageFileDataset
from ada_verona.database.machine_learning_model.onnx_network import ONNXNetwork
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler


@pytest.fixture
def network():
    return ONNXNetwork("./example_experiment/data/networks/mnist-net_256x2.onnx")


@pytest.fixture
def dataset():
    dataset = ImageFileDataset(image_folder=Path("./example_experiment/data/images"),
                                label_file=Path("./example_experiment/data/image_labels.csv"))
    return dataset


def test_sample_correct_predictions( network, dataset):
    sampler = PredictionsBasedSampler(sample_correct_predictions=True)
    # Mock the ONNX runtime session
    sampled_dataset = sampler.sample(network, dataset)
    selected_indices = [data_point.id for data_point in sampled_dataset]

    assert selected_indices == [ x.id for x in dataset._id_indices if x.id != 'mnist_train_80']


def test_sample_incorrect_predictions(network, dataset):
    sampler = PredictionsBasedSampler(sample_correct_predictions=False)

    sampled_dataset = sampler.sample(network, dataset)
    selected_indices = [data_point.id for data_point in sampled_dataset]

    assert selected_indices == ['mnist_train_80']
