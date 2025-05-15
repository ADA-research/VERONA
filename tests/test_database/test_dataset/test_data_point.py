import torch

from robustness_experiment_box.database.dataset.data_point import DataPoint


def test_to_dict():
    # Arrange
    data_point = DataPoint(
        id="dp1",
        label=1,
        data=torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    expected_dict = {
        "id": "dp1",
        "label": 1,
        "data": [[1.0, 2.0], [3.0, 4.0]]
    }

    # Act
    result = data_point.to_dict()

    # Assert
    assert result == expected_dict


def test_from_dict():
    # Arrange
    data_dict = {
        "id": "dp1",
        "label": 1,
        "data": [[1.0, 2.0], [3.0, 4.0]]
    }
    expected_data_point = DataPoint(
        id="dp1",
        label=1,
        data=torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )

    # Act
    result = DataPoint.from_dict(data_dict)

    # Assert
    assert result.id == expected_data_point.id
    assert result.label == expected_data_point.label
    assert torch.equal(result.data, expected_data_point.data)