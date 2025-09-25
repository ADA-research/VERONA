from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from typing_extensions import Self

from ada_verona.database.dataset.data_point import DataPoint
from ada_verona.database.dataset.experiment_dataset import ExperimentDataset


@dataclass
class IDIndex:
    """
    A class to represent an index and its corresponding ID for images.
    This distinction needs to be made because sometimes images have specific names and not just integer names.

    Attributes:
        index (int): The index of the data point.
        id (str): The identifier of the data point.
    """

    index: int
    id: str


class ImageFileDataset(ExperimentDataset):
    """
    A dataset class for loading images and their labels from files.
    """

    def __init__(self, image_folder: Path, label_file: Path, preprocessing: transforms.Compose = None) -> None:
        """
        Initialize the ImageFileDataset with the image folder, label file, and optional preprocessing.

        Args:
            image_folder (Path): The folder containing the images.
            label_file (Path): The file containing the labels.
            preprocessing (transforms.Compose, optional): The preprocessing transformations to apply to the images.
        """
        self.image_folder = image_folder
        self.label_file = label_file
        self.preprocessing = preprocessing

        self.image_data_df = self.merge_label_file_with_images(self.image_folder, self.label_file)

        self.data = self.get_labeled_image_list()

        self._id_indices = [IDIndex(x, self.data[x][0].name.split(".")[0]) for x in range(0, len(self.data))]

    def get_labeled_image_list(self) -> list[tuple[Path, int]]:
        """
        Get the list of labeled images.

        Returns:
            list[tuple[Path, int]]: The list of image paths and their corresponding labels.
        """
        data = [
            (self.image_folder / image_path, label)
            for image_path, label in zip(self.image_data_df.image, self.image_data_df.label, strict=False)
        ]
        return data

    def merge_label_file_with_images(self, image_folder: Path, image_label_file: Path) -> pd.DataFrame:
        """
        Merge the label file with the images in the image folder such that
        we have the label corresponding to the correct image.

        Args:
            image_folder (Path): The folder containing the images.
            image_label_file (Path): The file containing the labels.

        Returns:
            pd.DataFrame: The DataFrame containing the merged image paths and labels.
        """
        image_path_list = sorted(file.name for file in image_folder.iterdir())  #SORT FOR CONSISTENT LOADING

        image_path_df = pd.DataFrame(image_path_list, columns=["image"])
        image_label_df = pd.read_csv(image_label_file, index_col=0)

        return image_path_df.merge(image_label_df, how="left", on="image")

    def __len__(self) -> int:
        """
        Get the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return len(self._id_indices)

    def __getitem__(self, idx: int) -> DataPoint:
        """
        Get the data point at the specified index.

        Args:
            idx (int): The index of the data point.

        Returns:
            DataPoint: The data point at the specified index.
        """
        id_index = self._id_indices[idx]
        path, label = self.data[id_index.index]

        image = torch.load(path)

        if self.preprocessing:
            image = self.preprocessing(image)

        return DataPoint(id_index.id, label, image)

    def get_id_index_from_value(self, value: str) -> str:
        """
        Get the IDIndex object for the specified value.

        Args:
            value (str): The value to search for.

        Returns:
            str: The IDIndex object for the specified value.
        """
        for id_index in self._id_indices:
            if id_index.id == value:
                return id_index
        return -1

    def get_id_indices_from_value_list(self, value_list: list[str]) -> list[int]:
        """
        Get the list of IDIndex objects for the specified value list.

        Args:
            value_list (list[str]): The list of values to search for.

        Returns:
            list[int]: The list of IDIndex objects for the specified value list.
        """
        id_indices = []
        for value in value_list:
            id_index = self.get_id_index_from_value(value)
            id_indices.append(id_index)
        return id_indices

    def get_subset(self, values: list[str]) -> Self:
        """
        Get a subset of the dataset for the specified values.

        Args:
            values (list[str]): The list of values to get the subset for.

        Returns:
            Self: The subset of the dataset.
        """
        new_instance = ImageFileDataset(self.image_folder, self.label_file)

        new_instance._id_indices = self.get_id_indices_from_value_list(values)

        return new_instance
