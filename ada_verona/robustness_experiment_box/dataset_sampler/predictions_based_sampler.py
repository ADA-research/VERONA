import logging

import numpy as np
import onnxruntime as rt

from ada_verona.robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.robustness_experiment_box.database.datastructure.network import Network
from ada_verona.robustness_experiment_box.dataset_sampler.dataset_sampler import DatasetSampler


class PredictionsBasedSampler(DatasetSampler):
    """
    A sampler class that selects data points based on the predictions of a network.
    """

    def __init__(self, sample_correct_predictions: bool = True, sample_size: int | None = None, seed: int = 42) -> None:
        """
        Initialize the PredictionsBasedSampler with the given parameters.

        Args:
            sample_correct_predictions (bool, optional): Whether to sample data points with correct predictions. 
                Defaults to True as in the JAIR paper.
            sample_size (int | None, optional):Maximum number of samples to return. If None,returns all matching samples
                Defaults to None.
            seed (int, optional): Random seed for reproducible sampling when limiting sample size. Defaults to 42.
        """
        self.sample_correct_predictions = sample_correct_predictions
        self.sample_size = sample_size
        self.seed = seed

    def sample(self, network: Network, dataset: ExperimentDataset) -> ExperimentDataset:
        """
        Sample data points from the dataset based on the predictions of the network.

        Args:
            network (Network): The network to use for predictions.
            dataset (ExperimentDataset): The dataset to sample from.

        Returns:
            ExperimentDataset: The sampled dataset.
        """
        input_shape = network.get_input_shape()

        sess_opt = rt.SessionOptions()
        sess_opt.intra_op_num_threads = 1
        sess = rt.InferenceSession(str(network.path), sess_options=sess_opt)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        selected_indices = []

        for data_point in dataset:
            try:
                prediction_onnx = sess.run(
                    [label_name], {input_name: data_point.data.reshape(input_shape).detach().numpy()}
                )[0]
                predicted_label = np.argmax(prediction_onnx)
            except Exception as e:
                raise Exception(f"Opening inference session for network {network.path} failed with error: {e}") from e

            if self.sample_correct_predictions:
                if predicted_label == int(data_point.label):
                    selected_indices.append(data_point.id)
            else:
                if predicted_label != int(data_point.label):
                    selected_indices.append(data_point.id)

        # Apply size limiting if specified
        if self.sample_size is not None and len(selected_indices) > self.sample_size:
            logging.info(f"Limiting samples from {len(selected_indices)} to {self.sample_size} (seed: {self.seed})")
            # Use random sampling for reproducibility
            np.random.seed(self.seed)
            selected_indices = np.random.choice(selected_indices, self.sample_size, replace=False).tolist()
        else:
            logging.info(f"Selected {len(selected_indices)} data points (no size limiting applied)")

        return dataset.get_subset(selected_indices)

