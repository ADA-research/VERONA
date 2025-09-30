import torch

from ada_verona.database.dataset.data_point import DataPoint


def test_to_dict():
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

    result = data_point.to_dict()

    assert result == expected_dict


def test_from_dict():
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

    result = DataPoint.from_dict(data_dict)

    assert result.id == expected_data_point.id
    assert result.label == expected_data_point.label
    assert torch.equal(result.data, expected_data_point.data)