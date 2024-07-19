"""
Module defining the verification context. 

A verification context is one instance of a network, image and vnnlib property. 
vnnlib property already contains the epsilon. 
"""
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
from robustness_experiment_box.database.epsilon_status import EpsilonStatus

class VerificationContext:

    def __init__(self, network: Network, data_point: DataPoint, tmp_path: Path, save_epsilon_results: bool = True) -> None:   
        self.network = network
        self.data_point = data_point
        self.tmp_path = tmp_path
        self.save_epsilon_results = save_epsilon_results

        if save_epsilon_results and not self.tmp_path.exists():
            self.tmp_path.mkdir(parents=True)

    def get_dict_for_epsilon_result(self):
            return dict(network_path = self.network.path.resolve(), image_id = self.data_point.id, tmp_path = self.tmp_path.resolve())

    def save_vnnlib_property(self, vnnlib_property: VNNLibProperty) -> None:
        """
        Save the VNNLib property to a file in the temporary path.

        Args:
            vnnlib_property (VNNLibProperty): The VNNLib property object to be saved.
        """
        if not self.tmp_path.exists():
            self.tmp_path.mkdir(parents=True)
        save_path = self.tmp_path / f"{vnnlib_property.name}.vnnlib"

        with open(save_path, "w") as f:
            f.write(vnnlib_property.content)
        vnnlib_property.path = save_path

    def delete_tmp_path(self) -> None:
        """
        Delete the temporary path and its contents.
        """
        self.tmp_path.unlink()

    def save_status_list(self, epsilon_status_list: list[EpsilonStatus]) -> None:

        save_path = self.tmp_path / "epsilon_results.csv"
        data = [x.to_dict() for x in epsilon_status_list]
        df = pd.DataFrame(data=data)
        df.to_csv(save_path)

    def save_result(self, result: EpsilonStatus) -> None:
        if self.save_epsilon_results:
        
            result_df_path = self.tmp_path / "epsilons_df.csv"
            
            if result_df_path.exists():
                df = pd.read_csv(result_df_path, index_col=0)
                df.loc[len(df.index)] = result.to_dict()
            else:
                df = pd.DataFrame([result.to_dict()])
            df.to_csv(result_df_path)
    
    def to_dict(self):
        return { 
            'network': self.network.to_dict(),
            'data_point': self.data_point.to_dict(),
            'tmp_path': str(self.tmp_path),
            'save_epsilon_results': self.save_epsilon_results
        }

    @classmethod
    def from_dict(cls,data: dict):
        network = Network.from_dict(data['network'])
        data_point = DataPoint.from_dict(data['data_point'])
        tmp_path = Path(data['tmp_path'])
        save_epsilon_results=data['save_epsilon_results']
        return cls(network=network, data_point=data_point, tmp_path=tmp_path, save_epsilon_results= save_epsilon_results)


