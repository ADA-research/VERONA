from pathlib import Path
import pandas as pd
import torch
from typing_extensions import Self
import torchvision.transforms as transforms

from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset
from robustness_experiment_box.database.dataset.data_point import DataPoint

class ImageFileDataset(ExperimentDataset):

    def __init__(self, image_folder: Path, label_file: Path, preprocessing: transforms.Compose = None) -> None:
        self.image_folder = image_folder
        self.label_file = label_file
        self.preprocessing = preprocessing

        self.image_data_df = self.merge_label_file_with_images(self.image_folder, self.label_file)

        self.data = self.get_labeled_image_list()

        self._indices = [x for x in range(0,len(self.data))]


    def get_labeled_image_list(self) -> list[tuple[Path, int]]:
        data = [(self.image_folder / image_path, label) for image_path, label in zip(self.image_data_df.image, self.image_data_df.label)]
        return data
    
    def merge_label_file_with_images(self, image_folder: Path, image_label_file: Path):
        image_path_list = [file.name for file in image_folder.iterdir()]

        image_path_df = pd.DataFrame(image_path_list, columns=["image"])
        image_label_df = pd.read_csv(image_label_file, index_col=0)

        return image_path_df.merge(image_label_df, how="left", on="image")
    
    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx) -> DataPoint:

        index = self._indices[idx]
        path,label = self.data[index]

        image = torch.load(path)

        if self.preprocessing:
            image = self.preprocessing(image)

        return DataPoint(index, label, image)
    
    def get_subset(self, indices: list[int]) -> Self:
        new_instance = ImageFileDataset(self.image_folder, self.label_file)
        new_instance._indices = indices

        return new_instance

