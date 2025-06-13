import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from robustness_experiment_box.analysis.report_creator import ReportCreator
from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator

DEFAULT_RESULT_CSV_NAME = "result_df.csv"
PER_EPSILON_RESULT_CSV_NAME = "per_epsilon_results.csv"


class ExperimentRepository:
    """
    Database to handle all the paths to the different files used.
    """

    def __init__(self, base_path: Path, network_folder: Path) -> None:
        """
        Initialize the ExperimentRepository with the base path and network folder.

        Args:
            base_path (Path): The base path for the experiment repository.
            network_folder (Path): The folder containing the network files.
        """
        self.act_experiment_path = None
        self.base_path = base_path
        self.network_folder = network_folder

    def get_act_experiment_path(self) -> Path:
        """
        Get the path to the active experiment.

        Returns:
            Path: The path to the active experiment.

        Raises:
            Exception: If no experiment is loaded.
        """
        if self.act_experiment_path is not None:
            return self.act_experiment_path
        else:
            raise Exception("No experiment loaded")

    def get_results_path(self) -> Path:
        """
        Get the path to the results folder of the active experiment.

        Returns:
            Path: The path to the results folder.
        """
        return self.get_act_experiment_path() / "results"

    def get_tmp_path(self) -> Path:
        """
        Get the path to the temporary folder of the active experiment.

        Returns:
            Path: The path to the temporary folder.
        """
        return self.get_act_experiment_path() / "tmp"

    def initialize_new_experiment(self, experiment_name: str) -> None:
        """
        Initialize a new experiment with the given name.

        Args:
            experiment_name (str): The name of the experiment.

        Raises:
            Exception: If a directory with the same name already exists.
        """
        now = datetime.now()
        now_string = now.strftime("%d-%m-%Y+%H_%M")

        self.act_experiment_path = self.base_path / f"{experiment_name}_{now_string}"

        if os.path.exists(self.get_results_path()):
            raise Exception(
                "Error, there is already a directory with results with the same name,"
                "make sure no results will be overwritten"  
            )
        else:
            os.makedirs(self.get_results_path())
        os.makedirs(self.get_tmp_path())

    def load_experiment(self, experiment_name: str) -> None:
        """
        Load an existing experiment with the given name.

        Args:
            experiment_name (str): The name of the experiment.
        """
        self.act_experiment_path = self.base_path / experiment_name

    def save_configuration(self, data: dict) -> None:
        """
        Save the configuration data to a JSON file.

        Args:
            data (dict): The configuration data to save.
        """
        with open(self.get_act_experiment_path() / "configuration.json", "w") as outfile:
            json.dump(data, outfile)

    def get_network_list(self) -> list[Network]:
        """
        Get the list of networks from the network folder.

        Returns:
            list[Network]: The list of networks.
        """
        network_path_list = [file for file in self.network_folder.iterdir()]
        network_list = [Network(x) for x in network_path_list]
        return network_list

    def save_results(self, results: list[EpsilonValueResult]) -> None:
        """
        Save the list of epsilon value results to a CSV file.

        Args:
            results (list[EpsilonValueResult]): The list of epsilon value results to save.
        """
        result_df = pd.DataFrame([x.to_dict() for x in results])
        result_df.to_csv(self.get_results_path() / DEFAULT_RESULT_CSV_NAME)

    def save_result(self, result: EpsilonValueResult) -> None:
        """
        Save a single epsilon value result to the CSV file.

        Args:
            result (EpsilonValueResult): The epsilon value result to save.
        """
        result_df_path = self.get_results_path() / DEFAULT_RESULT_CSV_NAME
        if result_df_path.exists():
            df = pd.read_csv(result_df_path, index_col=0)
            df.loc[len(df.index)] = result.to_dict()
        else:
            df = pd.DataFrame([result.to_dict()])
        df.to_csv(result_df_path)

    def get_file_name(self, file: Path) -> str:
        """
        Get the name of the file without the extension.

        Args:
            file (Path): The file path.

        Returns:
            str: The name of the file without the extension.
        """
        return file.name.split(".")[0]

    def create_verification_context(
        self, network: Network, data_point: DataPoint, property_generator: PropertyGenerator
    ) -> VerificationContext:
        """
        Create a verification context for the given network, data point, and property generator.

        Args:
            network (Network): The network to verify.
            data_point (DataPoint): The data point to verify.
            property_generator (PropertyGenerator): The property generator to use.

        Returns:
            VerificationContext: The created verification context.
        """
        tmp_path = self.get_tmp_path() / f"{self.get_file_name(network.path)}" / f"image_{data_point.id}"
        return VerificationContext(network, data_point, tmp_path, property_generator)

    def get_result_df(self) -> pd.DataFrame:
        """
        Get the result DataFrame from the results CSV file.

        Returns:
            pd.DataFrame: The result DataFrame.

        Raises:
            Exception: If no result file is found.
        """
        result_df_path = self.get_results_path() / DEFAULT_RESULT_CSV_NAME
        if result_df_path.exists():
            df = pd.read_csv(result_df_path, index_col=0)
            df["network"] = df.network_path.str.split("/").apply(lambda x: x[-1]).apply(lambda x: x.split(".")[0])

            return df
        else:
            raise Exception(f"Error, no result file found at {result_df_path}")

    def get_per_epsilon_result_df(self) -> pd.DataFrame:
        """
        Get the per-epsilon result DataFrame from the temporary folder.

        Returns:
            pd.DataFrame: The per-epsilon result DataFrame.
        """
        per_epsilon_result_df_name = "epsilons_df.csv"
        df = pd.DataFrame()
        network_folders = [x for x in self.get_tmp_path().iterdir()]
        for network_folder in network_folders:
            images_folders = [x for x in network_folders[0].iterdir()]
            for image_folder in images_folders:
                t_df = pd.read_csv(image_folder / per_epsilon_result_df_name, index_col=0)
                t_df["network"] = network_folder.name
                t_df["image"] = image_folder.name
                df = pd.concat([df, t_df])
        return df

    def save_per_epsilon_result_df(self) -> None:
        """
        Save the per-epsilon result DataFrame to a CSV file.
        """
        per_epsilon_result_df = self.get_per_epsilon_result_df()
        per_epsilon_result_df.to_csv(self.get_results_path() / PER_EPSILON_RESULT_CSV_NAME)

    def save_plots(self) -> None:
        """
        Save the plots generated from the result DataFrame.
        """
        df = self.get_result_df()
        report_creator = ReportCreator(df)
        hist_figure = report_creator.create_hist_figure()
        hist_figure.savefig(self.get_results_path() / "hist_figure.png", bbox_inches="tight")

        boxplot = report_creator.create_box_figure()
        boxplot.savefig(self.get_results_path() / "boxplot.png", bbox_inches="tight")

        kde_figure = report_creator.create_kde_figure()
        kde_figure.savefig(self.get_results_path() / "kde_plot.png", bbox_inches="tight")

        ecdf_figure = report_creator.create_ecdf_figure()
        ecdf_figure.savefig(self.get_results_path() / "ecdf_plot.png", bbox_inches="tight")

    def save_verification_context_to_yaml(self, file_path: Path, verification_context: VerificationContext) -> Path:
        """
        Save the verification context to a YAML file.

        Args:
            file_path (Path): The path to save the YAML file.
            verification_context (VerificationContext): The verification context to save.

        Returns:
            Path: The path to the saved YAML file.
        """
        with open(file_path, "w") as file:
            yaml.dump(verification_context.to_dict(), file)
        return file_path

    def load_verification_context_from_yaml(self, file_path: Path) -> VerificationContext:
        """
        Load the verification context from a YAML file.

        Args:
            file_path (Path): The path to the YAML file.

        Returns:
            VerificationContext: The loaded verification context.
        """
        with open(file_path) as file:
            data = yaml.safe_load(file)
            return VerificationContext.from_dict(data)

    def cleanup_tmp_directory(self):
        """
        Delete the temporary folder of the active experiment.
        """
        
        tmp_path = self.get_tmp_path()
        if tmp_path.exists():
            for file in tmp_path.iterdir():
                file.unlink()
            tmp_path.rmdir()