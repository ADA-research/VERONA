import torch
import pandas as pd
import torchvision
from pathlib import Path

def create_image_folder(folder: Path, dataset: torchvision.datasets.MNIST):
    data = []
    for i in range(0,100):
        act_image = dataset[i][0]
        act_label = dataset[i][1]
        image_name = f"mnist_{i}.pt"
        torch.save(act_image, folder / f"images/{image_name}")
        data.append([image_name, act_label])
    df = pd.DataFrame(data,columns=[["image", "label"]])
    df.to_csv(folder / "images/image_labels.csv")