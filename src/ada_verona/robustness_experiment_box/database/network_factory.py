from pathlib import Path
from typing import Union

import pandas as pd

from ada_verona.robustness_experiment_box.database.base_network import BaseNetwork
from ada_verona.robustness_experiment_box.database.network import Network
from ada_verona.robustness_experiment_box.database.pytorch_network import PyTorchNetwork


class NetworkFactory:
    """
    Factory class for creating networks of different types.
    
    This class handles the creation of both ONNX and PyTorch networks
    based on configuration data.
    """

    @staticmethod
    def create_network_from_csv_row(row: pd.Series, networks_dir: Path) -> BaseNetwork:
        """
        Create a network from a CSV row.

        Args:
            row (pd.Series): A row from the networks CSV file.
            networks_dir (Path): The base directory containing network files.

        Returns:
            BaseNetwork: The created network.

        Raises:
            ValueError: If the network type is not supported or required fields are missing.
        """
        if "type" not in row:
            # Default to ONNX for backward compatibility
            network_type = "onnx"
        else:
            network_type = row["type"].lower()

        if network_type == "onnx":
            if "network_path" not in row:
                raise ValueError("ONNX network requires 'network_path' field")
            network_path = networks_dir / row["network_path"]
            return Network(path=network_path)
        
        elif network_type == "pytorch":
            if "architecture" not in row or "weights" not in row:
                raise ValueError("PyTorch network requires both 'architecture' and 'weights' fields")
            architecture_path = networks_dir / row["architecture"]
            weights_path = networks_dir / row["weights"]
            return PyTorchNetwork(architecture_path=architecture_path, weights_path=weights_path)
        
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

    @staticmethod
    def create_networks_from_csv(csv_path: Path, networks_dir: Path) -> list[BaseNetwork]:
        """
        Create a list of networks from a CSV file.

        Args:
            csv_path (Path): Path to the networks CSV file.
            networks_dir (Path): The base directory containing network files.

        Returns:
            list[BaseNetwork]: List of created networks.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the CSV format is invalid.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Networks CSV file not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"Networks CSV file is empty: {csv_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing networks CSV file: {e}")

        required_columns = ["name"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in networks CSV: {missing_columns}")

        networks = []
        for _, row in df.iterrows():
            try:
                network = NetworkFactory.create_network_from_csv_row(row, networks_dir)
                networks.append(network)
            except Exception as e:
                raise ValueError(f"Error creating network from row {row.to_dict()}: {e}")

        return networks

    @staticmethod
    def create_networks_from_directory(networks_dir: Path) -> list[BaseNetwork]:
        """
        Create networks from a directory (backward compatibility method).
        
        This method scans the directory for ONNX files and creates Network objects.

        Args:
            networks_dir (Path): The directory containing network files.

        Returns:
            list[BaseNetwork]: List of created networks.
        """
        networks = []
        for file_path in networks_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == ".onnx":
                networks.append(Network(path=file_path))
        return networks
