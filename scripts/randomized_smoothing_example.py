"""
Example usage of Randomized Smoothing with VERONA.

This example demonstrates how to use probabilistic certification.

Usage:
    python randomized_smoothing_example.py
"""

import logging
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

from robustness_experiment_box.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from robustness_experiment_box.database.experiment_repository import ExperimentRepository
from robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from robustness_experiment_box.epsilon_value_estimator.randomized_smoothing_estimator import (
    RandomizedSmoothingEstimator,
)
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from robustness_experiment_box.verification_module.randomized_smoothing_module import RandomizedSmoothingModule

# Configure logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - matches existing script structure
experiment_name = "randomized_smoothing_example"
experiment_repository_path = Path("../tests/test_experiment")
network_folder = Path("../tests/test_experiment/data/networks")

# Certification parameters
sigma = 0.25      # Noise level for smoothing
n0 = 100          # Samples for prediction
n = 10000         # Samples for certification
alpha = 0.001     # Confidence level
batch_size = 1000

def main():
    """Demonstrate randomized smoothing certification following VERONA patterns."""

    # Load dataset (same as existing scripts)
    logger.info("Loading MNIST dataset...")
    dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True,
        transform=transforms.ToTensor()
    )
    experiment_dataset = PytorchExperimentDataset(dataset)

    # For demonstration, we'll use a simple placeholder classifier
    # In practice, this would be your trained model
    class SimpleClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(784, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            return self.fc(x)

    base_classifier = SimpleClassifier()
    num_classes = 10

    # Create randomized smoothing module (replaces formal verifier)
    smoothing_module = RandomizedSmoothingModule(
        base_classifier=base_classifier,
        num_classes=num_classes,
        sigma=sigma
    )

    # Create estimator (replaces binary search estimator)
    estimator = RandomizedSmoothingEstimator(
        smoothing_module=smoothing_module,
        n0=n0,
        n=n,
        alpha=alpha,
        batch_size=batch_size
    )

    # Set up experiment repository (same as existing scripts)
    experiment_repository = ExperimentRepository(
        base_path=experiment_repository_path,
        network_folder=network_folder
    )

    # Initialize experiment (same as existing scripts)
    experiment_repository.initialize_new_experiment(experiment_name)

    # Save configuration (same as existing scripts)
    experiment_repository.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(experiment_repository_path),
            network_folder=str(network_folder),
            dataset=str(experiment_dataset),
            sigma=sigma,
            n0=n0,
            n=n,
            alpha=alpha,
            batch_size=batch_size,
        )
    )

    # Create property generator (same as existing scripts)
    property_generator = One2AnyPropertyGenerator()

    # Create sampler (same as existing scripts)
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)

    # Get network list (same as existing scripts)
    network_list = experiment_repository.get_network_list()
    logger.info(f"Found {len(network_list)} networks")

    # Main processing loop (same pattern as existing scripts)
    for network in network_list:
        logger.info(f"Processing network: {network.path}")

        # Sample data (same as existing scripts)
        sampled_data = dataset_sampler.sample(network, experiment_dataset)
        logger.info(f"Processing {len(sampled_data)} data points...")

        for data_point in sampled_data:
            logger.info(f"  Processing image ID: {data_point.id}, Label: {data_point.label}")

            # Create verification context (same as existing scripts)
            verification_context = experiment_repository.create_verification_context(
                network, data_point, property_generator
            )

            # Compute certified radius (same method call as existing scripts)
            epsilon_value_result = estimator.compute_epsilon_value(verification_context)

            # Save result (same as existing scripts)
            experiment_repository.save_result(epsilon_value_result)

            # Show detailed probabilistic result
            probabilistic_result = estimator.get_probabilistic_result(verification_context)

            logger.info("  Results:")
            logger.info(f"    Certified radius: {epsilon_value_result.epsilon:.4f}")
            logger.info(f"    Predicted class: {probabilistic_result.predicted_class}")
            logger.info(f"    Statistical confidence: {probabilistic_result.confidence:.4f}")
            logger.info(f"    Samples used: n0={probabilistic_result.n0}, n={probabilistic_result.n}")
            logger.info(f"    Certification time: {probabilistic_result.certification_time:.2f}s")

    # Save plots and finalize (same as existing scripts)
    experiment_repository.save_plots()

    logger.info("\n" + "="*60)
    logger.info("RANDOMIZED SMOOTHING CERTIFICATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Networks processed: {len(network_list)}")
    logger.info(f"Total certified radii computed: {len(sampled_data) * len(network_list)}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
