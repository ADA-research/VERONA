from pathlib import Path

import pandas as pd

from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.database.epsilon_status import EpsilonStatus
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator


class VerificationContext:
    """
    A class to represent the context for verification.
    This class saves all the relevant information for a verification run.
    """

    def __init__(
        self,
        network: Network,
        data_point: DataPoint,
        tmp_path: Path,
        property_generator: PropertyGenerator,
        save_epsilon_results: bool = True,
    ) -> None:
        """
        Initialize the VerificationContext with the given parameters.

        Args:
            network (Network): The network to be verified.
            data_point (DataPoint): The data point to be verified.
            tmp_path (Path): The temporary path for saving intermediate results.
            property_generator (PropertyGenerator): The property generator for creating verification properties.
            save_epsilon_results (bool, optional): Whether to save epsilon results. Defaults to True.
        """
        self.network = network
        self.data_point = data_point
        self.tmp_path = tmp_path
        self.property_generator = property_generator
        self.save_epsilon_results = save_epsilon_results

        if save_epsilon_results and not self.tmp_path.exists():
            self.tmp_path.mkdir(parents=True)

    def get_dict_for_epsilon_result(self) -> dict:
        """
        Get a dictionary representation of the epsilon result.

        Returns:
            dict: The dictionary representation of the epsilon result.
        """
        return dict(
            network_path=self.network.path.resolve(),
            image_id=self.data_point.id,
            original_label=self.data_point.label,
            tmp_path=self.tmp_path.resolve(),
            **self.property_generator.get_dict_for_epsilon_result(),
        )

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
        """
        Save the list of epsilon statuses to a CSV file.

        Args:
            epsilon_status_list (list[EpsilonStatus]): The list of epsilon statuses to save.
        """
        save_path = self.tmp_path / "epsilon_results.csv"
        data = [x.to_dict() for x in epsilon_status_list]
        df = pd.DataFrame(data=data)
        df.to_csv(save_path)

    def save_result(self, result: EpsilonStatus) -> None:
        """
        Save a single epsilon status result to the CSV file.

        Args:
            result (EpsilonStatus): The epsilon status result to save.
        """
        if self.save_epsilon_results:
            result_df_path = self.tmp_path / "epsilons_df.csv"
            if result_df_path.exists():
                df = pd.read_csv(result_df_path, index_col=0)
                df.loc[len(df.index)] = result.to_dict()
            else:
                df = pd.DataFrame([result.to_dict()])
            df.to_csv(result_df_path)

    def to_dict(self) -> dict:
        """
        Convert the VerificationContext to a dictionary.

        Returns:
            dict: The dictionary representation of the VerificationContext.
        """
        return {
            "network": self.network.to_dict(),
            "data_point": self.data_point.to_dict(),
            "tmp_path": str(self.tmp_path),
            "property_generator": self.property_generator.to_dict(),
            "save_epsilon_results": self.save_epsilon_results,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a VerificationContext from a dictionary.

        Args:
            data (dict): The dictionary containing the VerificationContext attributes.

        Returns:
            VerificationContext: The created VerificationContext.
        """
        network = Network.from_dict(data["network"])
        data_point = DataPoint.from_dict(data["data_point"])
        tmp_path = Path(data["tmp_path"])
        property_generator = PropertyGenerator.from_dict(data["property_generator"])
        save_epsilon_results = data["save_epsilon_results"]
        return cls(
            network=network,
            data_point=data_point,
            tmp_path=tmp_path,
            property_generator=property_generator,
            save_epsilon_results=save_epsilon_results,
        )
