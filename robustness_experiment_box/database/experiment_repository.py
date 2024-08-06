from pathlib import Path
import os
import pandas as pd
import json
from datetime import datetime
from torch.utils.data import Dataset
import yaml
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.analysis.report_creator import ReportCreator
from robustness_experiment_box.database.dataset.data_point import DataPoint

DEFAULT_RESULT_CSV_NAME = "result_df.csv"
PER_EPSILON_RESULT_CSV_NAME = "per_epsilon_results.csv"

class ExperimentRepository:
    "Database to handle all the paths to the different files used"

    def __init__(self, base_path: Path, network_folder: Path) -> None:

        self.act_experiment_path = None

        self.base_path = base_path
        self.network_folder = network_folder

    def get_act_experiment_path(self):
        if not self.act_experiment_path is None:
            return self.act_experiment_path
        else:
            raise Exception("No experiment loaded")
        
    def get_results_path(self):
        return self.get_act_experiment_path() / "results"
    
    def get_tmp_path(self):
        return self.get_act_experiment_path() / "tmp"

    def initialize_new_experiment(self, experiment_name: str):

        now = datetime.now()
        now_string = now.strftime("%d-%m-%Y+%H_%M")

        self.act_experiment_path = self.base_path / f"{experiment_name}_{now_string}"

        if os.path.exists(self.get_results_path()):
            raise Exception("Error, there is already a directory with results with the same name, make sure no results will be overwritten")
        else: 
            os.makedirs(self.get_results_path())
        os.makedirs(self.get_tmp_path())

    def load_experiment(self, experiment_name: str):
        self.act_experiment_path = self.base_path / experiment_name

    def save_configuration(self, data: dict) -> None:
        with open(self.get_act_experiment_path() / "configuration.json", "w") as outfile: 
            json.dump(data, outfile)
    
    def get_network_list(self) -> list[Network]:
        network_path_list = [file for file in self.network_folder.iterdir()]
        network_list = [Network(x) for x in network_path_list]
        return network_list
    
    def save_results(self, results: list[EpsilonValueResult]) -> None:
        result_df = pd.DataFrame([x.to_dict() for x in results])
        result_df.to_csv(self.get_results_path() / DEFAULT_RESULT_CSV_NAME)

    def save_result(self, result: EpsilonValueResult) -> None:
        result_df_path = self.get_results_path() / DEFAULT_RESULT_CSV_NAME
        if result_df_path.exists():
            df = pd.read_csv(result_df_path, index_col=0)
            df.loc[len(df.index)] = result.to_dict()
        else:
            df = pd.DataFrame([result.to_dict()])
        df.to_csv(result_df_path)

    def get_file_name(self, file: Path) -> str:
        return file.name.split(".")[0]

    def create_verification_context(self, network: Network, data_point: DataPoint) -> VerificationContext:
        tmp_path = self.get_tmp_path() / f"{self.get_file_name(network.path)}" / f"image_{data_point.id}"
        return VerificationContext(network, data_point, tmp_path)
    
    def get_result_df(self):
        result_df_path = self.get_results_path() / DEFAULT_RESULT_CSV_NAME
        if result_df_path.exists():
            df = pd.read_csv(result_df_path, index_col=0)
            df["network"] = df.network_path.str.split("/").apply(lambda x: x[-1]).apply(lambda x : x.split(".")[0])

            return df
        else:
            raise Exception(f"Error, no result file found at {result_df_path}")
        
    def get_per_epsilon_result_df(self):
        per_epsilon_result_df_name = "epsilons_df.csv"
        df = pd.DataFrame()
        network_folders = [x for x in self.get_tmp_path().iterdir()]
        for network_folder in network_folders:
            images_folders = [x for x in network_folders[0].iterdir()]
            for image_folder in images_folders:
                t_df = pd.read_csv(image_folder / per_epsilon_result_df_name, index_col = 0)
                t_df["network"] = network_folder.name
                t_df["image"] = image_folder.name
                df = pd.concat([df, t_df])
        return df
    
    def save_per_epsilon_result_df(self):
        per_epsilon_result_df = self.get_per_epsilon_result_df()
        per_epsilon_result_df.to_csv(self.get_results_path() / PER_EPSILON_RESULT_CSV_NAME)

    def save_plots(self):

        df = self.get_result_df()
        report_creator = ReportCreator(df)
        hist_figure = report_creator.create_hist_figure()
        hist_figure.savefig(self.get_results_path() / "hist_figure.png", bbox_inches='tight')

        boxplot = report_creator.create_box_figure()
        boxplot.savefig(self.get_results_path() / "boxplot.png", bbox_inches='tight')

        kde_figure = report_creator.create_kde_figure()
        kde_figure.savefig(self.get_results_path() / "kde_plot.png", bbox_inches='tight')

        ecdf_figure = report_creator.create_ecdf_figure()
        ecdf_figure.savefig(self.get_results_path() / "ecdf_plot.png", bbox_inches='tight')

    def save_verification_context_to_yaml(self, file_path: Path, verification_context: VerificationContext):
         with open(file_path, 'w') as file:
            yaml.dump(verification_context.to_dict(), file)

    def load_verification_context_from_yaml(self, file_path: Path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return VerificationContext.from_dict(data)



    

    
        


    
