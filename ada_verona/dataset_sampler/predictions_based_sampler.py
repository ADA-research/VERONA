from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.machine_learning_model.network import Network
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler


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

        selected_indices = []

        model = network.load_pytorch_model()

        for data_point in dataset:
            data = data_point.data.reshape(network.get_input_shape())
            output = model(data)

            _, predicted_label = output.max(1, keepdim=True)
            if self.sample_correct_predictions:
                if predicted_label == int(data_point.label):
                    selected_indices.append(data_point.id)
            else:
                if predicted_label != int(data_point.label):
                    selected_indices.append(data_point.id)


        return dataset.get_subset(selected_indices)

