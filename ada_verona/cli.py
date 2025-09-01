"""
Command-line interface for ada-verona.

This module provides a command-line interface for running robustness experiments
with ada-verona, including integration with the auto-verify plugin for formal verification.
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


def boolean_arg_type(value: str) -> str:
    """
    Custom argument type for boolean arguments.
    
    Args:
        value (str): String value to validate
        
    Returns:
        str: The validated string value
        
    Raises:
        argparse.ArgumentTypeError: If the value is not a valid boolean representation
    """
    if value.lower() in ("true", "false", "1", "0"):
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid boolean value. Use True/False or 1/0."
        )


def parse_boolean_arg(value: str) -> bool:
    """
    Parse a string boolean argument to a boolean value.
    
    Args:
        value (str): String representation of boolean ("True"/"False" or "1"/"0")
        
    Returns:
        bool: The parsed boolean value
        
    Raises:
        ValueError: If the value cannot be parsed as a boolean
    """
    # Handle case-insensitive comparison
    value_lower = value.lower()
    if value_lower in ("true", "1"):
        return True
    elif value_lower in ("false", "0"):
        return False
    else:
        # This should not happen if boolean_arg_type is used, but provide helpful error
        raise ValueError(f"Cannot parse '{value}' as boolean. Use True/False or 1/0.")


def build_transforms_from_args(args):
    """
    Build a list of transforms from command-line arguments.
    
    Args:
        args: Command-line arguments containing transform specifications
        
    Returns:
        list: List of torchvision transforms to apply sequentially
    """
    transforms_list = []
    
    # Handle deprecated --resize argument for backward compatibility
    if hasattr(args, 'resize') and args.resize:
        logging.warning("--resize is deprecated. Use --transforms Resize instead.")
        transforms_list.append(transforms.Resize(args.resize))
    
    # Process --transforms argument
    if hasattr(args, 'transforms') and args.transforms:
        for transform_name in args.transforms:
            transform_name = transform_name.lower()
            
            if transform_name == "resize":
                # Resize requires dimensions, use default if not specified
                if hasattr(args, 'resize') and args.resize:
                    transforms_list.append(transforms.Resize(args.resize))
                else:
                    # Default resize for common datasets
                    if hasattr(args, 'dataset') and args.dataset == "cifar10":
                        transforms_list.append(transforms.Resize((32, 32)))
                    else:
                        transforms_list.append(transforms.Resize((28, 28)))
                    logging.info(f"Applied default resize for {args.dataset}")
            
            elif transform_name == "totensor":
                transforms_list.append(transforms.ToTensor())
            
            elif transform_name == "normalize":
                # Apply normalization if mean/std are specified
                if hasattr(args, 'normalize_mean') and hasattr(args, 'normalize_std'):
                    if args.normalize_mean and args.normalize_std:
                        transforms_list.append(transforms.Normalize(args.normalize_mean, args.normalize_std))
                    else:
                        logging.warning("--normalize specified but --normalize-mean or --normalize-std missing")
                else:
                    logging.warning("--normalize specified but normalization parameters not provided")
            
            elif transform_name == "randomhorizontalflip":
                if hasattr(args, 'random_horizontal_flip') and args.random_horizontal_flip:
                    transforms_list.append(transforms.RandomHorizontalFlip(args.random_horizontal_flip))
                else:
                    transforms_list.append(transforms.RandomHorizontalFlip(0.5))
                    logging.info("Applied default RandomHorizontalFlip with p=0.5")
            
            elif transform_name == "randomrotation":
                if hasattr(args, 'random_rotation') and args.random_rotation:
                    transforms_list.append(transforms.RandomRotation(args.random_rotation))
                else:
                    transforms_list.append(transforms.RandomRotation(15))
                    logging.info("Applied default RandomRotation with range=15")
            
            elif transform_name == "colorjitter":
                if hasattr(args, 'color_jitter') and args.color_jitter:
                    transforms_list.append(transforms.ColorJitter(*args.color_jitter))
                else:
                    transforms_list.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.1))
                    logging.info("Applied default ColorJitter")
            
            elif transform_name == "grayscale":
                transforms_list.append(transforms.Grayscale())
            
            elif transform_name == "centercrop":
                if hasattr(args, 'center_crop') and args.center_crop:
                    transforms_list.append(transforms.CenterCrop(args.center_crop))
                else:
                    logging.warning("--centercrop specified but size not provided")
            
            else:
                logging.warning(f"Unknown transform: {transform_name}. Skipping.")
    
    # Always add ToTensor if not already present
    if not any(isinstance(t, type(transforms.ToTensor())) for t in transforms_list):
        transforms_list.append(transforms.ToTensor())
        logging.info("Added default ToTensor transform")
    
    return transforms_list


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="ADA-VERONA: Model Robustness Analysis Framework",
        epilog="""
ADA-VERONA is a framework for ML model robustness analysis.
It supports both empirical verification using adversarial attacks and formal verification
through the auto-verify plugin system.

For help, use: ada-verona --help
For command-specific help, use: ada-verona <command> --help
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # Disable default help to use our custom help
    )
    
    parser.add_argument(
        "--version", action="version", version=f"ada-verona {ada_verona.__version__}"
    )
    parser.add_argument(
        "--help", "-h", action="store_true",
        help="Show help and examples"
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
        help="CSV file containing custom labels (overrides default dataset labels)"
    )
    # Preprocessing options
    dataset_group.add_argument(
        "--train", type=boolean_arg_type, default="False",
        help="Use training split (True) or test split (False)"
    )
    dataset_group.add_argument(
        "--download", type=boolean_arg_type, default="True",
        help="Download dataset if not present"
    )
    dataset_group.add_argument(
        "--transforms", type=str, nargs="+", default=["ToTensor"],
        help="List of transforms to apply sequentially (e.g., Resize ToTensor Normalize)"
    )
    dataset_group.add_argument(
        "--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
        help="Resize images to WIDTH x HEIGHT (e.g., --resize 28 28) [DEPRECATED: use --transforms Resize instead]"
    )
    dataset_group.add_argument(
        "--normalize-mean", type=float, nargs="+", metavar="MEAN",
        help="Mean values for normalization (e.g., --normalize-mean 0.5 0.5 0.5 for RGB)"
    )
    dataset_group.add_argument(
        "--normalize-std", type=float, nargs="+", metavar="STD",
        help="Standard deviation values for normalization (e.g., --normalize-std 0.5 0.5 0.5 for RGB)"
    )
    dataset_group.add_argument(
        "--random-horizontal-flip", type=float, metavar="PROBABILITY",
        help="Probability for random horizontal flip (e.g., --random-horizontal-flip 0.5)"
    )
    dataset_group.add_argument(
        "--random-rotation", type=float, metavar="DEGREES",
        help="Random rotation range in degrees (e.g., --random-rotation 15)"
    )
    dataset_group.add_argument(
        "--color-jitter", type=float, nargs=4, metavar=("BRIGHTNESS", "CONTRAST", "SATURATION", "HUE"),
        help="Color jitter parameters (e.g., --color-jitter 0.2 0.2 0.2 0.1)"
    )
    dataset_group.add_argument(
        "--center-crop", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
        help="Center crop images to WIDTH x HEIGHT (e.g., --center-crop 224 224)"
    )
    
    # Verification options
    verify_group = run_parser.add_argument_group("Verification options")
    verify_group.add_argument(
        "--verifier", type=str, choices=["pgd", "fgsm", "auto-verify"],
        default="pgd", help="Verification method to use"
    )
    verify_group.add_argument(
        "--auto-verify-verifier", type=str,
        help="Name of auto-verify verifier to use (when --verifier=auto-verify)"
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
        "--sample-size", type=int, default=None,
        help="Number of samples per network (None = use all available samples)"
    )
    sample_group.add_argument(
        "--sample-correct", type=boolean_arg_type, default="True",
        help="Sample only correctly predicted inputs (True/False or 1/0, default: True)"
    )
    
    # Reproducibility options
    reproducibility_group = run_parser.add_argument_group("Reproducibility options")
    reproducibility_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Use PyTorch data command
    pytorch_parser = subparsers.add_parser("use-pytorch-data", help="Create robustness distribution on PyTorch dataset")
    
    # PyTorch dataset options
    pytorch_parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar10"], required=True,
        help="PyTorch dataset to use"
    )
    pytorch_parser.add_argument(
        "--name", type=str, default="pytorch_robustness_experiment",
        help="Name of the experiment"
    )
    pytorch_parser.add_argument(
        "--output", type=str, default="./experiment",
        help="Output directory for experiment results"
    )
    pytorch_parser.add_argument(
        "--networks", type=str, default="./experiment/networks",
        help="Directory containing network files (.onnx)"
    )
    pytorch_parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory for dataset storage/loading"
    )
    pytorch_parser.add_argument(
        "--train", type=boolean_arg_type, default="False",
        help="Use training split (True) or test split (False)"
    )
    pytorch_parser.add_argument(
        "--download", type=boolean_arg_type, default="True",
        help="Download dataset if not present"
    )
    pytorch_parser.add_argument(
        "--custom-labels", type=str,
        help="CSV file containing custom labels (overrides default dataset labels)"
    )
    pytorch_parser.add_argument(
        "--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
        help="Resize images to WIDTH x HEIGHT (e.g., --resize 28 28) [DEPRECATED: use --transforms Resize instead]"
    )
    pytorch_parser.add_argument(
        "--transforms", type=str, nargs="+", default=["ToTensor"],
        help="List of transforms to apply sequentially (e.g., Resize ToTensor Normalize)"
    )
    pytorch_parser.add_argument(
        "--normalize-mean", type=float, nargs="+", metavar="MEAN",
        help="Mean values for normalization (e.g., --normalize-mean 0.5 0.5 0.5 for RGB)"
    )
    pytorch_parser.add_argument(
        "--normalize-std", type=float, nargs="+", metavar="STD",
        help="Standard deviation values for normalization (e.g., --normalize-std 0.5 0.5 0.5 for RGB)"
    )
    pytorch_parser.add_argument(
        "--random-horizontal-flip", type=float, metavar="PROBABILITY",
        help="Probability for random horizontal flip (e.g., --random-horizontal-flip 0.5)"
    )
    pytorch_parser.add_argument(
        "--random-rotation", type=float, metavar="DEGREES",
        help="Random rotation range in degrees (e.g., --random-rotation 15)"
    )
    pytorch_parser.add_argument(
        "--color-jitter", type=float, nargs=4, metavar=("BRIGHTNESS", "CONTRAST", "SATURATION", "HUE"),
        help="Color jitter parameters (e.g., --color-jitter 0.2 0.2 0.2 0.1)"
    )
    pytorch_parser.add_argument(
        "--center-crop", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
        help="Center crop images to WIDTH x HEIGHT (e.g., --center-crop 224 224)"
    )
    pytorch_parser.add_argument(
        "--verifier", type=str, choices=["pgd", "fgsm"], default="fgsm",
        help="Verification method to use"
    )
    pytorch_parser.add_argument(
        "--pgd-iterations", type=int, default=10,
        help="Number of iterations for PGD attack"
    )
    pytorch_parser.add_argument(
        "--pgd-step-size", type=float, default=0.01,
        help="Step size for PGD attack"
    )

    pytorch_parser.add_argument(
        "--property", type=str, choices=["one2any", "one2one"],
        default="one2any", help="Property type for verification"
    )
    pytorch_parser.add_argument(
        "--epsilons", type=float, nargs="+", 
        default=[0.001, 0.005, 0.05, 0.08],
        help="List of epsilon values to search"
    )
    pytorch_parser.add_argument(
        "--sample-correct", type=boolean_arg_type, default="True",
        help="Sample only correctly predicted inputs (True/False or 1/0, default: True)"
    )
    pytorch_parser.add_argument(
        "--target-class", type=int, default=1,
        help="Target class for one2one property (default: 1)"
    )
    
    # Reproducibility options
    pytorch_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Help command
    subparsers.add_parser("help", help="Show help and examples")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument(
        "--verifiers", action="store_true",
        help="List available verification methods"
    )
    list_parser.add_argument(
        "--auto-verify", action="store_true",
        help="List auto-verify integration status"
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
            logging.error("Auto-verify not available. Please install auto-verify to use formal verification.")
            sys.exit(1)
            
        if not args.auto_verify_verifier:
            logging.error("--auto-verify-verifier must be specified when using auto-verify.")
            logging.info("Available auto-verify verifiers:")
            if ada_verona.AUTO_VERIFY_VERIFIERS:
                for verifier in sorted(ada_verona.AUTO_VERIFY_VERIFIERS):
                    logging.info(f"  • {verifier}")
            else:
                logging.info("  No verifiers found. Install with: auto-verify install <verifier>")
            sys.exit(1)
            
        if args.auto_verify_verifier not in ada_verona.AUTO_VERIFY_VERIFIERS:
            logging.error(f"Auto-verify verifier '{args.auto_verify_verifier}' not found.")
            logging.info(f"Available verifiers: {', '.join(ada_verona.AUTO_VERIFY_VERIFIERS)}")
            sys.exit(1)
        
        # Create the auto-verify verifier
        verifier = ada_verona.create_auto_verify_verifier(
            verifier_name=args.auto_verify_verifier,
            timeout=args.timeout
        )
        
        if verifier is None:
            logging.error(f"Failed to create auto-verify verifier '{args.auto_verify_verifier}'.")
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
    # Parse boolean arguments
    train_bool = args.train.lower() == "true" if hasattr(args, 'train') else False
    download_bool = args.download.lower() == "true" if hasattr(args, 'download') else True
    
    # Build transforms
    data_transforms = build_transforms_from_args(args)
    
    if args.dataset == "mnist":
        torch_dataset = torchvision.datasets.MNIST(
            root=args.data_dir, train=train_bool, download=download_bool, transform=transforms.Compose(data_transforms)
        )
        dataset = PytorchExperimentDataset(dataset=torch_dataset)
        
        # Apply custom labels if provided
        if args.custom_labels:
            custom_labels_path = Path(args.custom_labels)
            if not custom_labels_path.exists() or not custom_labels_path.is_file():
                logging.error(f"Custom labels file {custom_labels_path} does not exist or is not a file.")
                sys.exit(1)
            logging.info(f"Using custom labels from: {custom_labels_path}")
            # Note: PytorchExperimentDataset doesn't support custom labels, so we log a warning
            logging.warning("Custom labels not supported for PyTorch datasets. Labels from original dataset will be used.")
        
        return dataset
    
    elif args.dataset == "cifar10":
        torch_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=train_bool, download=download_bool, transform=transforms.Compose(data_transforms)
        )
        dataset = PytorchExperimentDataset(dataset=torch_dataset)
        
        # Apply custom labels if provided
        if args.custom_labels:
            custom_labels_path = Path(args.custom_labels)
            if not custom_labels_path.exists() or not custom_labels_path.is_file():
                logging.error(f"Custom labels file {custom_labels_path} does not exist or is not a file.")
                sys.exit(1)
            logging.info(f"Using custom labels from: {custom_labels_path}")
            # Note: PytorchExperimentDataset doesn't support custom labels, so we log a warning
            logging.warning("Custom labels not supported for PyTorch datasets. Labels from original dataset will be used.")
        
        return dataset
    
    elif args.dataset == "custom":
        if not args.custom_images or not args.custom_labels:
            logging.error("--custom-images and --custom-labels must be provided for custom dataset.")
            sys.exit(1)
            
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
        root=args.data_dir, train=train_bool, download=download_bool, transform=transforms.Compose(data_transforms)
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
    torch.manual_seed(args.seed)
    logging.info(f"Random seed set to: {args.seed}")
    
    # Log experiment configuration
    logging.info("=== ADA-VERONA Robustness Experiment ===")
    logging.info(f"Experiment name: {args.name}")
    logging.info(f"Output directory: {args.output}")
    logging.info(f"Network directory: {args.networks}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Verifier: {args.verifier}")
    if args.verifier == "pgd":
        logging.info(f"PGD iterations: {args.pgd_iterations}")
        logging.info(f"PGD step size: {args.pgd_step_size}")
    elif args.verifier == "auto-verify":
        logging.info(f"Auto-verify verifier: {args.auto_verify_verifier}")
    logging.info(f"Epsilon values: {args.epsilons}")
    logging.info(f"Sample size: {args.sample_size}")
    
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
    try:
        sample_correct_bool = parse_boolean_arg(args.sample_correct)
        logging.info(f"Sample correct predictions: {sample_correct_bool}")
    except ValueError as e:
        logging.error(f"Invalid --sample-correct value: {e}")
        sys.exit(1)
        
    dataset_sampler = PredictionsBasedSampler(
        sample_correct_predictions=sample_correct_bool,
        sample_size=args.sample_size,
        seed=args.seed
    )
    
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
            sample_correct=sample_correct_bool,
            pgd_iterations=args.pgd_iterations,
            pgd_step_size=args.pgd_step_size,
            seed=args.seed,
            train=args.train,
            download=args.download,
            transforms=args.transforms,
            normalize_mean=args.normalize_mean if hasattr(args, 'normalize_mean') else None,
            normalize_std=args.normalize_std if hasattr(args, 'normalize_std') else None,
            random_horizontal_flip=args.random_horizontal_flip if hasattr(args, 'random_horizontal_flip') else None,
            random_rotation=args.random_rotation if hasattr(args, 'random_rotation') else None,
            color_jitter=args.color_jitter if hasattr(args, 'color_jitter') else None,
            center_crop=args.center_crop if hasattr(args, 'center_crop') else None,
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


def run_pytorch_experiment(args):
    """
    Run a PyTorch dataset robustness experiment based on command-line arguments.
    
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
    torch.manual_seed(args.seed)
    logging.info(f"Random seed set to: {args.seed}")
    
    # Log experiment configuration
    logging.info("=== ADA-VERONA PyTorch Dataset Robustness Experiment ===")
    logging.info(f"Experiment name: {args.name}")
    logging.info(f"Output directory: {args.output}")
    logging.info(f"Network directory: {args.networks}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Verifier: {args.verifier}")
    if args.verifier == "pgd":
        logging.info(f"PGD iterations: {args.pgd_iterations}")
        logging.info(f"PGD step size: {args.pgd_step_size}")
    logging.info(f"Property type: {args.property}")
    logging.info(f"Epsilon values: {args.epsilons}")
    logging.info(f"Sample size: {args.sample_size}")
    logging.info(f"Train split: {args.train}")
    logging.info(f"Download: {args.download}")
    if args.resize:
        logging.info(f"Resize: {args.resize}")
    if hasattr(args, 'transforms') and args.transforms:
        logging.info(f"Transforms: {args.transforms}")
    if hasattr(args, 'normalize_mean') and args.normalize_mean:
        logging.info(f"Normalize mean: {args.normalize_mean}")
    if hasattr(args, 'normalize_std') and args.normalize_std:
        logging.info(f"Normalize std: {args.normalize_std}")
    
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
    if args.property == "one2one":
        property_generator = One2OnePropertyGenerator(target_class=args.target_class)
    else:
        property_generator = One2AnyPropertyGenerator()
    
    # Create verifier
    if args.verifier == "pgd":
        verifier = AttackEstimationModule(attack=PGDAttack(
            number_iterations=args.pgd_iterations,
            step_size=args.pgd_step_size
        ))
    else:  # fgsm
        verifier = AttackEstimationModule(attack=FGSMAttack())
    
    logging.info(f"Using verifier: {verifier.name}")
    
    # Create epsilon value estimator
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=args.epsilons.copy(),
        verifier=verifier
    )
    
    # Create dataset sampler
    try:
        sample_correct_bool = parse_boolean_arg(args.sample_correct)
        logging.info(f"Sample correct predictions: {sample_correct_bool}")
    except ValueError as e:
        logging.error(f"Invalid --sample-correct value: {e}")
        sys.exit(1)
        
    dataset_sampler = PredictionsBasedSampler(
        sample_correct_predictions=sample_correct_bool,
        sample_size=args.sample_size,
        seed=args.seed
    )
    
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
            pgd_iterations=args.pgd_iterations,
            pgd_step_size=args.pgd_step_size,
            seed=args.seed,
            train=args.train,
            download=args.download,
            resize=args.resize,
            sample_correct=sample_correct_bool,
            target_class=args.target_class if args.property == "one2one" else None,
            transforms=args.transforms,
            normalize_mean=args.normalize_mean if hasattr(args, 'normalize_mean') else None,
            normalize_std=args.normalize_std if hasattr(args, 'normalize_std') else None,
            random_horizontal_flip=args.random_horizontal_flip if hasattr(args, 'random_horizontal_flip') else None,
            random_rotation=args.random_rotation if hasattr(args, 'random_rotation') else None,
            color_jitter=args.color_jitter if hasattr(args, 'color_jitter') else None,
            center_crop=args.center_crop if hasattr(args, 'center_crop') else None,
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


def _handle_help():
    """Show comprehensive help and examples."""
    help_text = """
ADA-VERONA HELP
=============================

ADA-VERONA is a framework for model robustness analysis.
It supports both empirical verification using adversarial attacks and formal verification
through the auto-verify plugin system.

SUPPORTED DATASETS
-----------------
• mnist      - MNIST handwritten digit dataset
• cifar10    - CIFAR-10 image classification dataset  
• custom     - Custom dataset with images and labels

SUPPORTED VERIFICATION METHODS
-----------------------------
• pgd        - Projected Gradient Descent attack (empirical)
• fgsm       - Fast Gradient Sign Method attack (empirical)
• auto-verify - Formal verification via auto-verify plugin

SUPPORTED PROPERTY TYPES
-----------------------
• one2any    - One input to any other class (default)
• one2one    - One input to specific target class

CORE COMMANDS
-------------

1. RUN ROBUSTNESS EXPERIMENT
   ada-verona run [options]
   
   Examples:
   • ada-verona run --networks ./models --dataset mnist
   • ada-verona run --networks ./models --verifier pgd --epsilons 0.001 0.01 0.05
   • ada-verona run --networks ./models --verifier auto-verify --auto-verify-verifier abcrown
   • ada-verona run --networks ./models --dataset cifar10 --sample-size 20 --sample-correct
   
   Key Options:
   --networks <path>              Directory with network files (.onnx) [REQUIRED]
   --name <name>                  Experiment name (default: robustness_experiment)
   --output <path>                Output directory (default: ./experiment)
   --dataset {mnist,cifar10,custom} Dataset to use (default: mnist)
   --verifier {pgd,fgsm,auto-verify} Verification method (default: pgd)
   --property {one2any,one2one}   Property type (default: one2any)
   --epsilons <values>            List of epsilon values (default: [0.001, 0.005, 0.01, 0.05])
   --sample-size <number>         Samples per network (default: 10)
   --sample-correct               Sample only correctly predicted inputs

2. USE PYTORCH DATASET EXPERIMENT
   ada-verona use-pytorch-data [options]
   
   Examples:
   • ada-verona use-pytorch-data --dataset mnist --networks ./models
   • ada-verona use-pytorch-data --dataset cifar10 --verifier fgsm --property one2one
   • ada-verona use-pytorch-data --dataset mnist --transforms Resize ToTensor Normalize
   
   Key Options:
   --dataset {mnist,cifar10}      PyTorch dataset to use [REQUIRED]
   --networks <path>              Directory with network files (.pt) [REQUIRED]
   --verifier {pgd,fgsm}          Verification method (default: fgsm)
   --property {one2any,one2one}   Property type (default: one2any)
   --target-class <number>        Target class for one2one property (default: 1)

3. LIST AVAILABLE COMPONENTS
   ada-verona list [options]
   
   Examples:
   • ada-verona list                              # List all components
   • ada-verona list --verifiers                 # List verification methods
   • ada-verona list --auto-verify               # Check auto-verify status
   
   Options:
   --verifiers                    List available verification methods
   --auto-verify                  List auto-verify integration status

4. SHOW COMPREHENSIVE HELP
   ada-verona help
   
   Shows this comprehensive help information.

DATASET OPTIONS
---------------

For the 'run' command, you can customize dataset behavior:

• --data-dir <path>              Dataset storage directory (default: ./experiment/data)
• --train {True,False}           Use training split (default: False)
• --download {True,False}        Download dataset if not present (default: True)
• --custom-images <path>          Custom image directory (for custom dataset)
• --custom-labels <path>          Custom labels CSV file (overrides default dataset labels)

DATA PREPROCESSING OPTIONS
-------------------------

Transform pipeline for data preprocessing:

• --transforms <list>             List of transforms (default: ["ToTensor"])
• --resize <width> <height>       Resize images [DEPRECATED: use --transforms Resize]
• --normalize-mean <values>       Mean values for normalization
• --normalize-std <values>        Standard deviation values for normalization
• --random-horizontal-flip <p>    Probability for random horizontal flip
• --random-rotation <degrees>     Random rotation range in degrees
• --color-jitter <b> <c> <s> <h> Color jitter parameters
• --center-crop <width> <height>  Center crop images

Available transforms: Resize, ToTensor, Normalize, RandomHorizontalFlip, 
                     RandomRotation, ColorJitter, Grayscale, CenterCrop

TRANSFORM CHAINING
-----------------

Transforms are applied sequentially in the order specified. You can chain multiple
transforms together using the --transforms argument:

Examples:
• Basic preprocessing: --transforms Resize ToTensor
• With normalization: --transforms Resize ToTensor Normalize
• Data augmentation: --transforms RandomHorizontalFlip RandomRotation ToTensor
• Full pipeline: --transforms Resize RandomHorizontalFlip ColorJitter ToTensor Normalize

Transform order matters! Common patterns:
1. Resize first (if needed)
2. Data augmentation (RandomHorizontalFlip, RandomRotation, ColorJitter)
3. ToTensor (converts to tensor format)
4. Normalize last (applied to tensor data)

Note: ToTensor is automatically added if not present in your transform list.

VERIFICATION OPTIONS
-------------------

• --verifier {pgd,fgsm,auto-verify} Verification method
• --auto-verify-verifier <name>   Auto-verify verifier name
• --timeout <seconds>             Verification timeout (default: 300)
• --property {one2any,one2one}     Property type for verification

Attack-specific options (for pgd):
• --pgd-iterations <number>       Number of PGD iterations (default: 10)
• --pgd-step-size <value>         PGD step size (default: 0.01)

Note: PGD parameters are available in both 'run' and 'use-pytorch-data' commands.

EXPERIMENT CONFIGURATION
-----------------------

• --name <name>                   Experiment name
• --output <path>                 Output directory for results
• --networks <path>               Directory with network files (.onnx)
• --epsilons <values>             List of epsilon values to search
• --sample-size <number>          Number of samples per network (None = use all available)
• --sample-correct {True,False,1,0} Sample only correctly predicted inputs (default: True)

REPRODUCIBILITY
---------------

• --seed <number>                 Random seed for reproducibility (default: 42)

AUTO-VERIFY INTEGRATION
----------------------

When using auto-verify for formal verification:

• Install auto-verify: pip install auto-verify
• Install verifiers: auto-verify install <verifier>
• Use in ada-verona: --verifier auto-verify --auto-verify-verifier <name>

Supported auto-verify verifiers:
• abcrown    - Neural network verification tool
• nnenum     - Neural network enumeration tool
• ovalbab    - Neural network verification tool
• verinet    - Neural network verification tool

EXPERIMENT FOLDER STRUCTURE
--------------------------

Default structure (created automatically):
```
experiment/
|-- data/
|   |-- labels.csv
|   |-- images/
|       |-- mnist_0.npy
|       |-- mnist_1.npy
|       |-- ...
|-- networks/
|   |-- mnist-net_256x2.onnx
|   |-- mnist-net_256x4.onnx
|   |-- ...
```

You must provide ONNX network files in the networks directory.

EXAMPLES
--------

Basic experiment with PGD attack:
```bash
ada-verona run --networks ./models --name pgd_experiment --verifier pgd --epsilons 0.001 0.005 0.01
```

Using auto-verify with specific verifier:
```bash
ada-verona run --networks ./models --name formal_verification \
  --verifier auto-verify --auto-verify-verifier abcrown --timeout 600
```

Customizing dataset and sampling:
```bash
ada-verona run --networks ./models --dataset cifar10 --sample-size 20 --sample-correct True
ada-verona run --networks ./models --dataset cifar10 --sample-size 20 --sample-correct False
ada-verona run --networks ./models --dataset cifar10 --sample-size 20 --sample-correct 1
ada-verona run --networks ./models --dataset cifar10 --sample-size 20 --sample-correct 0
```

Using custom labels:
```bash
ada-verona run --networks ./models --dataset mnist --custom-labels ./my_labels.csv
ada-verona use-pytorch-data --dataset cifar10 --networks ./models --custom-labels ./custom_labels.csv
```

Reproducible experiments:
```bash
ada-verona run --networks ./models --seed 123 --sample-size 5
```

Sampling options examples:
```bash
# Sample only correctly predicted inputs (default behavior)
ada-verona run --networks ./models --sample-correct True

# Sample only incorrectly predicted inputs
ada-verona run --networks ./models --sample-correct False

# Using numeric values
ada-verona run --networks ./models --sample-correct 1  # Same as True
ada-verona run --networks ./models --sample-correct 0  # Same as False
```

PyTorch dataset experiment with transforms:
```bash
ada-verona use-pytorch-data --dataset mnist --networks ./models \
  --transforms Resize ToTensor Normalize --normalize-mean 0.5 --normalize-std 0.5
```

One-to-one property verification:
```bash
ada-verona use-pytorch-data --dataset mnist --networks ./models --property one2one --target-class 5
```

TROUBLESHOOTING
---------------

• Check available components: ada-verona list
• Check auto-verify status: ada-verona list --auto-verify
• Verify network files: Ensure .onnx files are in networks directory
• Check dataset availability: Ensure dataset can be downloaded/accessed
• Monitor experiment progress: Check output directory for logs and results

GETTING HELP
------------

• ada-verona --help                    # This comprehensive help
• ada-verona <command> --help         # Command-specific help
• ada-verona --version                 # Show version information

For more information, visit the project documentation or repository.
"""
    print(help_text)


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
        print("\nAuto-verify integration status:")
        print(f"  Available: {ada_verona.HAS_AUTO_VERIFY}")
        
        if ada_verona.HAS_AUTO_VERIFY:
            print("\nAvailable auto-verify verifiers:")
            if ada_verona.AUTO_VERIFY_VERIFIERS:
                for verifier in sorted(ada_verona.AUTO_VERIFY_VERIFIERS):
                    print(f"  • {verifier}")
            else:
                print("  No verifiers found. Install verifiers with: auto-verify install <verifier>")
            print("\nNote: Auto-verify is managed separately via the auto-verify CLI.")
        else:
            print("  Auto-verify not available. Install with: pip install auto-verify")
    
    # If no specific component was requested, show general info
    if not (args.verifiers or args.auto_verify):
        print("\nADA-VERONA Components:")
        print("  • Verification methods: use --verifiers to list")
        print("  • Auto-verify integration: use --auto-verify to check")
        print("\nFor auto-verify management, use the auto-verify CLI directly:")
        print("  • auto-verify install <verifier>")
        print("  • auto-verify list")
        print("  • auto-verify --help")


def main():
    """Main entry point for the CLI."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    # Check if --help was used (either explicitly or implicitly when no command given)
    if args.help or not args.command:
        _handle_help()
        return
    
    try:
        if args.command == "run":
            run_experiment(args)
        elif args.command == "use-pytorch-data":
            run_pytorch_experiment(args)
        elif args.command == "list":
            list_components(args)
        elif args.command == "help":
            _handle_help()
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