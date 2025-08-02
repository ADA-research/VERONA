"""
Command-line interface for ada-verona.

This module provides a command-line interface for running robustness experiments
with ada-verona, including integration with auto-verify.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

import ada_verona
from ada_verona.robustness_experiment_box.database.dataset.image_file_dataset import ImageFileDataset
from ada_verona.robustness_experiment_box.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from ada_verona.robustness_experiment_box.database.experiment_repository import ExperimentRepository
from ada_verona.robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
from ada_verona.robustness_experiment_box.verification_module.attacks.fgsm_attack import FGSMAttack
from ada_verona.robustness_experiment_box.verification_module.attacks.pgd_attack import PGDAttack
from ada_verona.robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.robustness_experiment_box.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)
from ada_verona.robustness_experiment_box.verification_module.verification_module import VerificationModule


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="ADA-VERONA: Neural Network Robustness Analysis Framework"
    )
    
    parser.add_argument(
        "--version", action="version", version=f"ada-verona {ada_verona.__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run experiment command
    run_parser = subparsers.add_parser("run", help="Run robustness experiment")
    
    # Experiment configuration
    run_parser.add_argument(
        "--name", type=str, default="robustness_experiment",
        help="Name of the experiment"
    )
    run_parser.add_argument(
        "--output", type=str, default="./experiment",
        help="Output directory for experiment results"
    )
    run_parser.add_argument(
        "--networks", type=str, default="./experiment/networks",
        help="Directory containing network files (.onnx)"
    )
    
    # Dataset options
    dataset_group = run_parser.add_argument_group("Dataset options")
    dataset_group.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar10", "custom"],
        default="mnist", help="Dataset to use"
    )
    dataset_group.add_argument(
        "--data-dir", type=str, default="./experiment/data",
        help="Directory for dataset storage/loading"
    )
    dataset_group.add_argument(
        "--custom-images", type=str,
        help="Directory containing custom image files (for custom dataset)"
    )
    dataset_group.add_argument(
        "--custom-labels", type=str,
        help="CSV file containing labels (for custom dataset)"
    )
    
    # Verification options
    verify_group = run_parser.add_argument_group("Verification options")
    verify_group.add_argument(
        "--verifier", type=str, choices=["pgd", "fgsm", "auto-verify"],
        default="pgd", help="Verification method to use"
    )
    verify_group.add_argument(
        "--auto-verify-name", type=str,
        help="Name of auto-verify verifier to use (requires auto-verify installed)"
    )
    verify_group.add_argument(
        "--auto-verify-venv", type=str,
        help="Name of auto-verify venv to use (requires auto-verify installed)"
    )
    verify_group.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout for verification in seconds"
    )
    verify_group.add_argument(
        "--property", type=str, choices=["one2any", "one2one"],
        default="one2any", help="Property type for verification"
    )
    
    # Attack parameters
    attack_group = run_parser.add_argument_group("Attack parameters (for attack-based verification)")
    attack_group.add_argument(
        "--pgd-iterations", type=int, default=10,
        help="Number of iterations for PGD attack"
    )
    attack_group.add_argument(
        "--pgd-step-size", type=float, default=0.01,
        help="Step size for PGD attack"
    )
    
    # Epsilon search options
    epsilon_group = run_parser.add_argument_group("Epsilon search options")
    epsilon_group.add_argument(
        "--epsilons", type=float, nargs="+", 
        default=[0.001, 0.005, 0.01, 0.05],
        help="List of epsilon values to search"
    )
    
    # Sample options
    sample_group = run_parser.add_argument_group("Sampling options")
    sample_group.add_argument(
        "--sample-size", type=int, default=10,
        help="Number of samples per network"
    )
    sample_group.add_argument(
        "--sample-correct", action="store_true",
        help="Sample only correctly predicted inputs"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument(
        "--verifiers", action="store_true",
        help="List available verifiers"
    )
    list_parser.add_argument(
        "--auto-verify", action="store_true",
        help="List available auto-verify verifiers"
    )
    list_parser.add_argument(
        "--auto-verify-venvs", action="store_true",
        help="List available auto-verify virtual environments"
    )
    
    return parser


def create_verifier(args) -> VerificationModule:
    """
    Create a verifier based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        VerificationModule: The configured verifier
    """
    if args.verifier == "pgd":
        return AttackEstimationModule(
            attack=PGDAttack(
                number_iterations=args.pgd_iterations,
                step_size=args.pgd_step_size
            )
        )
    elif args.verifier == "fgsm":
        return AttackEstimationModule(attack=FGSMAttack())
    elif args.verifier == "auto-verify":
        if not ada_verona.HAS_AUTO_VERIFY:
            logging.error("Auto-verify not installed. Please install auto-verify first.")
            sys.exit(1)
            
        # Check if a specific verifier name was provided
        if not args.auto_verify_name and not args.auto_verify_venv:
            logging.error("Either --auto-verify-name or --auto-verify-venv must be specified when using auto-verify.")
            sys.exit(1)
            
        # If venv name is specified, try to find the corresponding verifier
        if args.auto_verify_venv:
            # Import auto-verify to access venv information
            try:
                from autoverify.cli.install.venv_installers.venv_install import VENV_VERIFIER_DIR
                
                # Check if the specified venv exists
                venv_path = VENV_VERIFIER_DIR / args.auto_verify_venv
                if not venv_path.exists():
                    logging.error(f"Auto-verify venv '{args.auto_verify_venv}' not found.")
                    logging.info("Available venvs:")
                    
                    if VENV_VERIFIER_DIR.exists():
                        venvs = [d.name for d in VENV_VERIFIER_DIR.iterdir() if d.is_dir()]
                        for venv in sorted(venvs):
                            logging.info(f"  • {venv}")
                    else:
                        logging.info("  No venvs found.")
                    
                    sys.exit(1)
                
                # Use the venv name as the verifier name
                verifier_name = args.auto_verify_venv
                
            except ImportError:
                logging.error("Failed to import autoverify. Please ensure it's installed correctly.")
                sys.exit(1)
        else:
            # Use the specified verifier name
            verifier_name = args.auto_verify_name
            
            # Check if the verifier is available
            if verifier_name not in ada_verona.AUTO_VERIFY_VERIFIERS:
                logging.error(f"Auto-verify verifier '{verifier_name}' not found.")
                logging.info(f"Available verifiers: {ada_verona.AUTO_VERIFY_VERIFIERS}")
                sys.exit(1)
        
        # Create the verifier
        verifier = ada_verona.create_auto_verify_verifier(
            verifier_name=verifier_name,
            timeout=args.timeout
        )
        
        if verifier is None:
            logging.error(f"Failed to create auto-verify verifier '{verifier_name}'.")
            sys.exit(1)
            
        return verifier
    
    # Default fallback
    logging.warning(f"Unknown verifier '{args.verifier}', falling back to PGD attack.")
    return AttackEstimationModule(attack=PGDAttack(number_iterations=10, step_size=0.01))


def load_dataset(args):
    """
    Load dataset based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        PytorchExperimentDataset: The loaded dataset
    """
    if args.dataset == "mnist":
        torch_dataset = torchvision.datasets.MNIST(
            root=args.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        return PytorchExperimentDataset(dataset=torch_dataset)
    
    elif args.dataset == "cifar10":
        torch_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        return PytorchExperimentDataset(dataset=torch_dataset)
    
    elif args.dataset == "custom":
        if not args.custom_images or not args.custom_labels:
            logging.error("--custom-images and --custom-labels must be provided for custom dataset.")
            sys.exit(1)
            
        # Use the provided custom images and labels directly
        custom_images_path = Path(args.custom_images)
        custom_labels_path = Path(args.custom_labels)
        
        if not custom_images_path.exists() or not custom_images_path.is_dir():
            logging.error(f"Custom images directory {custom_images_path} does not exist or is not a directory.")
            sys.exit(1)
            
        if not custom_labels_path.exists() or not custom_labels_path.is_file():
            logging.error(f"Custom labels file {custom_labels_path} does not exist or is not a file.")
            sys.exit(1)
            
        return ImageFileDataset(image_folder=custom_images_path, label_file=custom_labels_path)
    
    # Default fallback
    logging.warning(f"Unknown dataset '{args.dataset}', falling back to MNIST.")
    torch_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    return PytorchExperimentDataset(dataset=torch_dataset)


def run_experiment(args):
    """
    Run a robustness experiment based on command-line arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Log experiment configuration
    logging.info("=== ADA-VERONA Robustness Experiment ===")
    logging.info(f"Experiment name: {args.name}")
    logging.info(f"Output directory: {args.output}")
    logging.info(f"Network directory: {args.networks}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Verifier: {args.verifier}")
    if args.verifier == "auto-verify":
        if args.auto_verify_venv:
            logging.info(f"Auto-verify venv: {args.auto_verify_venv}")
        else:
            logging.info(f"Auto-verify verifier: {args.auto_verify_name}")
    logging.info(f"Epsilon values: {args.epsilons}")
    
    
    # Load dataset
    dataset = load_dataset(args)
    logging.info("Dataset loaded.")
    
    # Create experiment repository
    experiment_repository = ExperimentRepository(
        base_path=output_path,
        network_folder=Path(args.networks)
    )
    
    # Ensure networks directory exists and contains network files
    networks_path = Path(args.networks)
    if not networks_path.exists():
        logging.info(f"Creating networks directory: {args.networks}")
        networks_path.mkdir(parents=True, exist_ok=True)
        logging.error(f"No network files found in {args.networks}. Please add .onnx network files to this directory.")
        logging.info("Experiment setup is complete, but cannot run without network files.")
        sys.exit(1)
    
    # Check if the networks directory contains any .onnx files
    onnx_files = list(networks_path.glob("*.onnx"))
    if not onnx_files:
        logging.error(f"No .onnx network files found in {args.networks}.")
        logging.info("Please add network files to continue with the experiment.")
        sys.exit(1)
    
    # Create property generator
    property_generator = One2AnyPropertyGenerator() if args.property == "one2any" else One2OnePropertyGenerator()
    
    # Create verifier
    verifier = create_verifier(args)
    logging.info(f"Using verifier: {verifier.name}")
    
    # Create epsilon value estimator
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=args.epsilons.copy(),
        verifier=verifier
    )
    
    # Create dataset sampler
    dataset_sampler = PredictionsBasedSampler(
        sample_correct_predictions=args.sample_correct
    )
    
    # Note: PredictionsBasedSampler doesn't support number_of_samples parameter
    # The sample size will be determined by the correct/incorrect predictions
    
    # Initialize experiment
    experiment_repository.initialize_new_experiment(args.name)
    
    # Save configuration
    experiment_repository.save_configuration(
        dict(
            experiment_name=args.name,
            experiment_repository_path=str(output_path),
            network_folder=str(args.networks),
            dataset=str(args.dataset),
            verifier=args.verifier,
            property_type=args.property,
            epsilon_list=[str(x) for x in args.epsilons],
            sample_size=args.sample_size,
            sample_correct=args.sample_correct,
        )
    )
    
    # Run experiment
    logging.info("Starting robustness distribution computation...")
    
    network_list = experiment_repository.get_network_list()
    failed_networks = []
    
    for network in network_list:
        try:
            logging.info(f"Processing network: {network.path.name}")
            sampled_data = dataset_sampler.sample(network, dataset)
            
        except Exception as e:
            logging.error(f"Failed to sample data for network: {network} with error: {e}")
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
    logging.info(f"Results saved to {output_path}")


def list_components(args):
    """
    List available components based on command-line arguments.
    
    Args:
        args: Command-line arguments
    """
    if args.verifiers:
        print("\nAvailable verification methods:")
        print("  • pgd - Projected Gradient Descent attack")
        print("  • fgsm - Fast Gradient Sign Method attack")
        if ada_verona.HAS_AUTO_VERIFY:
            print("  • auto-verify - Formal verification via auto-verify")
        else:
            print("  • auto-verify - [NOT AVAILABLE] Install auto-verify to enable")
            
    if args.auto_verify:
        print("\nAuto-verify status:")
        print(f"  Installed: {ada_verona.HAS_AUTO_VERIFY}")
        
        if ada_verona.HAS_AUTO_VERIFY:
            print("\nAvailable auto-verify verifiers:")
            if ada_verona.AUTO_VERIFY_VERIFIERS:
                for verifier in sorted(ada_verona.AUTO_VERIFY_VERIFIERS):
                    print(f"  • {verifier}")
            else:
                print("  No verifiers found. Install verifiers with: auto-verify install <verifier>")
        else:
            print("  Auto-verify not installed. Install with: pip install auto-verify")
            
    if args.auto_verify_venvs:
        print("\nAuto-verify virtual environments:")
        try:
            from autoverify.cli.install.venv_installers.venv_install import VENV_VERIFIER_DIR
            
            if VENV_VERIFIER_DIR.exists():
                venvs = [d.name for d in VENV_VERIFIER_DIR.iterdir() if d.is_dir()]
                if venvs:
                    for venv in sorted(venvs):
                        print(f"  • {venv}")
                        
                        # Try to get more details about the venv
                        tool_dir = VENV_VERIFIER_DIR / venv / "tool"
                        if tool_dir.exists():
                            try:
                                import subprocess

                                from autoverify.util.env import cwd
                                with cwd(tool_dir):
                                    result = subprocess.run(
                                        ["git", "rev-parse", "--short", "HEAD"],
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                    commit = result.stdout.strip()
                                    
                                    result = subprocess.run(
                                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                    branch = result.stdout.strip()
                                    
                                    print(f"    - Branch: {branch}")
                                    print(f"    - Commit: {commit}")
                            except Exception:
                                pass
                else:
                    print("  No virtual environments found.")
            else:
                print("  Virtual environment directory not found.")
                
        except ImportError:
            print("  Auto-verify not installed. Install with: pip install auto-verify")
    
    # If no specific component was requested, show general info
    if not (args.verifiers or args.auto_verify or args.auto_verify_venvs):
        print("\nADA-VERONA Components:")
        print("  • Verification methods: use --verifiers to list")
        print("  • Auto-verify status: use --auto-verify to check")
        print("  • Auto-verify environments: use --auto-verify-venvs to list")


def main():
    """Main entry point for the CLI."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "run":
            run_experiment(args)
        elif args.command == "list":
            list_components(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 