import pandas as pd
import pytest
import torch

from ada_verona.database.dataset.image_file_dataset import ImageFileDataset


@pytest.fixture
def image_file_dataset(tmp_path, mocker):
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    label_file = tmp_path / "labels.csv"

    # Create mock images
    for i in range(3):
        (image_folder / f"image_{i}.pt").write_bytes(b"mock_image_data")

    # Create mock label file
    label_data = pd.DataFrame({"image": [f"image_{i}.pt" for i in range(3)], "label": [0, 1, 2]})
    label_data.to_csv(label_file, index=False)
   
    # Mock torch.load to return a tensor
    mocker.patch("torch.load", return_value=torch.tensor([1.0, 2.0, 3.0]))
    
    #try out what happens when we mock the transform
    mock_transform = mocker.Mock()
    mock_transform.return_value = torch.tensor([0.5])
    dataset = ImageFileDataset(image_folder=image_folder, label_file=label_file, preprocessing=mock_transform)

    dummy_image = torch.tensor([1.0])
    
    result = dataset.preprocessing(dummy_image)

    mock_transform.assert_called_once_with(dummy_image)
    assert torch.equal(result, torch.tensor([0.5]))

    #Assert whether folders and files exist
    assert image_folder.exists()
    assert label_file.exists()
    assert label_data.equals(pd.read_csv(label_file))
    assert len(list(image_folder.iterdir())) == 3
    assert len(label_data) == 3
    
    #return imagefiledataset with the preprocessing
    return dataset



def test_len(image_file_dataset):
    dataset_length = len(image_file_dataset)

    assert dataset_length == 3


def test_getitem(image_file_dataset):
    data_point = image_file_dataset[1]
    
    assert data_point.id == "image_1"
    assert data_point.label == 1
    assert torch.equal(data_point.data, torch.tensor([0.5]))


def test_merge_label_file_with_images(image_file_dataset):
    merged_df = image_file_dataset.merge_label_file_with_images(
        image_file_dataset.image_folder, image_file_dataset.label_file
    )

    assert len(merged_df) == 3
    assert "image" in merged_df.columns
    assert "label" in merged_df.columns


def test_get_id_index_from_value(image_file_dataset):
    id_index = image_file_dataset.get_id_index_from_value("image_1")
    id_index_missing = image_file_dataset.get_id_index_from_value("image_4")
   
    assert id_index.id == "image_1"
    assert id_index.index == 1
    assert id_index_missing == -1



def test_get_id_indices_from_value_list(image_file_dataset):
    id_indices = image_file_dataset.get_id_indices_from_value_list(["image_0", "image_2"])

    assert len(id_indices) == 2
    assert id_indices[0].id == "image_0"
    assert id_indices[1].id == "image_2"


def test_get_subset(image_file_dataset):
    subset = image_file_dataset.get_subset(["image_0", "image_2"])
    
    assert len(subset) == 2
    assert subset[0].id == "image_0"
    assert subset[1].id == "image_2"


def test_get_id_index_from_value_missing(image_file_dataset):
    result = image_file_dataset.get_id_index_from_value("does_not_exist")
    assert result == -1
    
    

def test_getitem_without_preprocessing(tmp_path, mocker):
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    label_file = tmp_path / "labels.csv"

    (image_folder / "image_0.pt").write_bytes(b"mock_image_data")
    pd.DataFrame({"image": ["image_0.pt"], "label": [0]}).to_csv(label_file, index=False)

    mocker.patch("torch.load", return_value=torch.tensor([1.0, 2.0, 3.0]))

    dataset = ImageFileDataset(image_folder=image_folder, label_file=label_file, preprocessing=None)
    data_point = dataset[0]

    assert torch.equal(data_point.data, torch.tensor([1.0, 2.0, 3.0]))
