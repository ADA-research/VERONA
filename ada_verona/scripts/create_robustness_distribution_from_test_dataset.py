#!/usr/bin/env python3
"""
Unified script for creating robustness distributions from test datasets using different verifiers.
Supports abcrown, verinet, nnenum, and ovalbab verifiers.
"""

import argparse
import logging
from pathlib import Path

import ada_verona
from ada_verona.robustness_experiment_box.database.dataset.image_file_dataset import ImageFileDataset
from ada_verona.robustness_experiment_box.database.experiment_repository import ExperimentRepository
from ada_verona.robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create robustness distribution from test dataset using specified verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_robustness_distribution_from_test_dataset.py --verifier abcrown 
  python create_robustness_distribution_from_test_dataset.py --verifier verinet --timeout 30
  python create_robustness_distribution_from_test_dataset.py --verifier nnenum --epsilon-list 0.001 0.01 0.1
        """,
    )

    parser.add_argument(
        "--verifier",
        required=True,
        choices=["abcrown", "verinet", "nnenum", "ovalbab"],
        help="Verifier to use for robustness analysis",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for verification (default: 10)",
    )

    parser.add_argument(
        "--experiment-name",
        help="Custom experiment name (default: auto_verify_{verifier}_test)",
    )

    parser.add_argument(
        "--experiment-repository-path",
        type=Path,
        default=Path("../example_experiment"),
        help="Path to experiment repository (default: ../example_experiment)",
    )

    parser.add_argument(
        "--network-folder",
        type=Path,
        default=Path("../example_experiment/networks"),
        help="Path to network folder (default: ../example_experiment/networks)",
    )

    parser.add_argument(
        "--image-folder",
        type=Path,
        default=Path("../example_experiment/data/images"),
        help="Path to image folder (default: ../example_experiment/data/images)",
    )

    parser.add_argument(
        "--image-label-file",
        type=Path,
        default=Path("../example_experiment/data/image_labels.csv"),
        help="Path to image label file (default: ../example_experiment/data/image_labels.csv)",
    )

    parser.add_argument(
        "--epsilon-list",
        nargs="+",
        type=float,
        default=[0.001, 0.005],
        help="List of epsilon values for robustness analysis (default: 0.001 0.005)",
    )

    parser.add_argument(
        "--sample-correct-predictions",
        action="store_true",
        help="Sample correct predictions instead of incorrect ones (default: False)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def validate_verifier_availability(verifier_name):
    """Validate that the specified verifier is available."""
    if not ada_verona.HAS_AUTO_VERIFY:
        raise RuntimeError("Auto-verify is not available. Please install auto-verify package.")

    if verifier_name not in ada_verona.AUTO_VERIFY_VERIFIERS:
        raise RuntimeError(
            f"{verifier_name.title()} verifier is not available. "
            f"Available verifiers: {ada_verona.AUTO_VERIFY_VERIFIERS}"
        )


def create_verifier(verifier_name, timeout):
    """Create verifier using plugin architecture."""
    verifier = ada_verona.create_auto_verify_verifier(verifier_name, timeout=timeout)
    if verifier is None:
        raise RuntimeError(f"Failed to create {verifier_name.title()} verifier through plugin system.")
    return verifier


def main():
    """Main function to run the robustness distribution creation."""
    args = parse_arguments()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=log_level)

    # Validate verifier availability
    validate_verifier_availability(args.verifier)

    # Create verifier
    verifier = create_verifier(args.verifier, args.timeout)

    # Set experiment name
    experiment_name = args.experiment_name or f"auto_verify_{args.verifier}_test"

    logging.info(f"Starting robustness analysis with {args.verifier} verifier")
    logging.info(f"Experiment name: {experiment_name}")
    logging.info(f"Timeout: {args.timeout} seconds")
    logging.info(f"Epsilon values: {args.epsilon_list}")

    # Create dataset and database
    dataset = ImageFileDataset(image_folder=args.image_folder, label_file=args.image_label_file)
    file_database = ExperimentRepository(base_path=args.experiment_repository_path, network_folder=args.network_folder)

    # Initialize experiment
    file_database.initialize_new_experiment(experiment_name)

    # Save configuration
    file_database.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(args.experiment_repository_path),
            network_folder=str(args.network_folder),
            dataset=str(dataset),
            timeout=args.timeout,
            epsilon_list=[str(x) for x in args.epsilon_list],
            verifier=args.verifier,
        )
    )

    # Create components
    property_generator = One2AnyPropertyGenerator()
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=args.epsilon_list.copy(), verifier=verifier
    )
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=args.sample_correct_predictions)

    # Process networks
    network_list = file_database.get_network_list()
    logging.info(f"Processing {len(network_list)} networks")

    for network in network_list:
        logging.info(f"Processing network: {network}")
        sampled_data = dataset_sampler.sample(network, dataset)

        for data_point in sampled_data:
            verification_context = file_database.create_verification_context(network, data_point, property_generator)
            epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)
            file_database.save_result(epsilon_value_result)

    # Save plots
    file_database.save_plots()
    logging.info("Robustness analysis completed successfully")


if __name__ == "__main__":
    main()


    