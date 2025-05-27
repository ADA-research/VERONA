import numpy as np
import onnxruntime as rt

from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.dataset_sampler.dataset_sampler import DatasetSampler


class PredictionsBasedSampler(DatasetSampler):
    """
    A sampler class that selects data points based on the predictions of a network.
    """

    def __init__(self, sample_correct_predictions: bool = True) -> None:
        """
        Initialize the PredictionsBasedSampler with the given parameter.

        Args:
            sample_correct_predictions (bool, optional): Whether to sample data points with correct predictions. 
            Defaults to True as in the JAIR paper.
        """
        self.sample_correct_predictions = sample_correct_predictions

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

        return dataset.get_subset(selected_indices)

