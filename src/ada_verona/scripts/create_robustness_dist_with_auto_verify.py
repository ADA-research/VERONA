"""
Example script demonstrating auto-verify integration with ada-verona.

This script shows how to use formal verification tools from auto-verify
seamlessly within the ada-verona framework through the plugin system.
"""

import logging
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

import ada_verona
from ada_verona.robustness_experiment_box.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from ada_verona.robustness_experiment_box.database.experiment_repository import ExperimentRepository
from ada_verona.robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)

torch.manual_seed(0)
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def create_distribution_with_auto_verify(verifier_name: str = "nnenum"):
    """
    Create robustness distribution using auto-verify verifiers.
    
    Args:
        verifier_name: Name of the auto-verify verifier to use
    """
    logging.info("=== ADA-VERONA + Auto-Verify Integration Example ===")
    
    # Check auto-verify availability
    if not ada_verona.HAS_AUTO_VERIFY:
        logging.error("Auto-verify not detected! Please install auto-verify in the same environment.")
        logging.info("Available verifiers would be limited to attacks only.")
        return
    
    logging.info(f"Auto-verify detected! Available verifiers: {ada_verona.AUTO_VERIFY_VERIFIERS}")
    
    # Setup experiment parameters
    epsilon_list = [0.001, 0.005, 0.01, 0.02]  # Start with smaller epsilons for formal verification
    experiment_repository_path = Path("../tests/test_experiment")
    network_folder = Path("../tests/test_experiment/data/networks")
    
    # Setup dataset
    torch_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    dataset = PytorchExperimentDataset(dataset=torch_dataset)
    
    # Initialize experiment repository
    experiment_repository = ExperimentRepository(
        base_path=experiment_repository_path, 
        network_folder=network_folder
    )
    
    # Create auto-verify verifier through plugin system
    logging.info(f"Creating auto-verify verifier: {verifier_name}")
    
    try:
        auto_verify_module = ada_verona.create_auto_verify_verifier(
            verifier_name=verifier_name,
            timeout=300,  # 5 minutes timeout
            # Add any verifier-specific kwargs here
            # cpu_gpu_allocation=(0, 1, -1),  # Example for resource allocation
        )
        
        if auto_verify_module is None:
            logging.error(f"Failed to create auto-verify verifier '{verifier_name}'")
            logging.info(f"Available verifiers: {ada_verona.list_auto_verify_verifiers()}")
            return
            
        logging.info(f"Successfully created verifier module: {auto_verify_module.name}")
        
    except Exception as e:
        logging.error(f"Error creating auto-verify verifier: {e}")
        return
    
    # Setup experiment components
    experiment_name = f"auto_verify_{verifier_name}"
    property_generator = One2AnyPropertyGenerator()
    
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), 
        verifier=auto_verify_module
    )
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
    
    # Initialize experiment
    experiment_repository.initialize_new_experiment(experiment_name)
    experiment_repository.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(experiment_repository_path),
            network_folder=str(network_folder),
            dataset=str(dataset),
            verifier=verifier_name,
            verifier_type="auto_verify",
            epsilon_list=[str(x) for x in epsilon_list],
        )
    )
    
    # Run experiment
    logging.info("Starting robustness distribution computation with formal verification...")
    
    network_list = experiment_repository.get_network_list()
    failed_networks = []
    
    for network in network_list:
        try:
            sampled_data = dataset_sampler.sample(network, dataset)
            logging.info(f"Processing network: {network.path.name}")
            
        except Exception as e:
            logging.info(f"Failed to sample data for network: {network} with error: {e}")
            failed_networks.append(network)
            continue
            
        for data_point in sampled_data:
            try:
                verification_context = experiment_repository.create_verification_context(
                    network, data_point, property_generator
                )
                
                logging.info(f"Computing epsilon value for data point {data_point.id}")
                epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)
                
                experiment_repository.save_result(epsilon_value_result)
                logging.info(f"Epsilon result: {epsilon_value_result.epsilon}")
                
            except Exception as e:
                logging.error(f"Error processing data point {data_point.id}: {e}")
                continue
    
    # Save results and plots
    experiment_repository.save_plots()
    logging.info(f"Experiment completed. Failed networks: {failed_networks}")
    logging.info("Results saved to experiment repository.")


def demonstrate_plugin_capabilities():
    """Demonstrate the plugin system capabilities."""
    print("\nADA-VERONA Plugin System Demo")
    print("=" * 50)
    
    print(f"Auto-verify available: {ada_verona.HAS_AUTO_VERIFY}")
    print(f"PyAutoAttack available: {ada_verona.HAS_AUTOATTACK}")
    
    if ada_verona.HAS_AUTO_VERIFY:
        print("\nAvailable auto-verify verifiers:")
        for verifier in ada_verona.AUTO_VERIFY_VERIFIERS:
            print(f"  • {verifier}")
            
        print("\nUsage example:")
        print("  verifier = ada_verona.create_auto_verify_verifier('nnenum', timeout=600)")
        print("  # Use verifier in BinarySearchEpsilonValueEstimator or other components")
    else:
        print("\nAuto-verify not available - only attack-based verification available")
        print("   To enable formal verification, install auto-verify in the same environment")


if __name__ == "__main__":
    # Demonstrate plugin capabilities
    demonstrate_plugin_capabilities()
    
    # Run example with formal verification if available
    if ada_verona.HAS_AUTO_VERIFY and ada_verona.AUTO_VERIFY_VERIFIERS:
        # Use the first available verifier for demo
        verifier_name = ada_verona.AUTO_VERIFY_VERIFIERS[0]
        print(f"\nRunning example with {verifier_name}...")
        create_distribution_with_auto_verify(verifier_name)
    else:
        print("\Install auto-verify to run formal verification examples!")
        print("Example: pip install auto-verify") 