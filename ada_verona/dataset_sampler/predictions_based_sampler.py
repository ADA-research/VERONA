import torch

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.machine_learning_model.network import Network
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler


class PredictionsBasedSampler(DatasetSampler):
    """
    A sampler class that selects data points based on the predictions of a network.
    """

    def __init__(self, sample_correct_predictions: bool = True, top_k: int = 1) -> None:
        """
        Initialize the PredictionsBasedSampler with the given parameter.

        Args:
            sample_correct_predictions (bool, optional): Whether to sample data points with correct predictions. 
            Defaults to True as in the JAIR paper.
            top_k: Number of top scores to take into account for checking the correct prediction.
        """
        self.sample_correct_predictions = sample_correct_predictions
        self.top_k = top_k

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = network.load_pytorch_model().to(device)
        model.eval() 

        for data_point in dataset:
            data = data_point.data.reshape(network.get_input_shape())
            data = data.to(device) 

            with torch.no_grad():  
                output = model(data)

            _, predicted_labels = torch.topk(output, self.top_k)
            predicted_labels = predicted_labels.cpu()  
            
            if self.sample_correct_predictions:
                if int(data_point.label) in predicted_labels:
                    selected_indices.append(data_point.id)
            else:
                if int(data_point.label) not in predicted_labels:
                    selected_indices.append(data_point.id)

        return dataset.get_subset(selected_indices)

