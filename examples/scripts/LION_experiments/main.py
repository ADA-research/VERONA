# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import logging
import torch
import argparse
torch.manual_seed(0)
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm as timm
import torch
import torchvision
import torchvision.transforms as transforms
from onnx2torch import convert
from torch.utils.data import Subset

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.database.machine_learning_model.pytorch_network import PyTorchNetwork
from ada_verona.database.machine_learning_model.torch_model_wrapper import TorchModelWrapper
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from ada_verona.verification_module.attack_estimation_module import AttackEstimationModule
from ada_verona.verification_module.attacks.auto_attack_wrapper import AutoAttackWrapper
from ada_verona.verification_module.attacks.fgsm_attack import FGSMAttack
from ada_verona.verification_module.attacks.pgd_attack import PGDAttack
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator


def check_verified_instances(model_name, attack_method,indice):
    pattern = f"/scratch-shared/abosman/data/results_LION/{model_name}-{attack_method}--{indice}*"
    matches = glob.glob(pattern)
    
    if not matches:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    latest_file = max(matches, key=os.path.getmtime)

    print(f"Loading: {latest_file}")
    return pd.read_csv(latest_file).image_id.tolist()



def run_verification(model_name, attack_method,indice,batch_size):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device)

    network = PyTorchNetwork(model, [1, 3, 224,224], model_name)

    verified = check_verified_instances(model_name, attack_method,indice)
    
    epsilon_list = np.arange(0.00, 0.4, 0.0001)
    experiment_repository_path = Path("/scratch-shared/abosman/data/results_LION")
    torch_dataset = torchvision.datasets.ImageNet(
    root="/scratch-shared/abosman/data/data_LION",
    split="val",
    transform=transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
)

    dataset = PytorchExperimentDataset(dataset=torch_dataset)
    dataset = dataset.get_subset(range((indice*batch_size), ((indice*batch_size)+batch_size)))
    
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=None)

    experiment_name = f"{model_name}-{attack_method}--{indice}"
    property_generator = One2AnyPropertyGenerator()

    if attack_method == "fgsm":
        verifier = AttackEstimationModule(attack=FGSMAttack(), top_k=5)
    elif attack_method == "pgd":
        verifier = AttackEstimationModule(attack=PGDAttack(number_iterations=40), top_k=5)
    elif attack_method == "autoattack":
        verifier = AttackEstimationModule(attack=AutoAttackWrapper(), top_k=5)

    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), verifier=verifier
    )
    
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True, top_k=5)
    experiment_repository.initialize_new_experiment(experiment_name)
    experiment_repository.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(experiment_repository_path),
            dataset=str(dataset),
            epsilon_list=[str(x) for x in epsilon_list],
        )
    )

    sampled_data = dataset_sampler.sample(network, dataset)

    verified = set(verified)
    for data_point in sampled_data:
        if data_point.id in verified:
            continue
        
        verification_context = experiment_repository.create_verification_context(network, data_point, property_generator)

        epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

        experiment_repository.save_result(epsilon_value_result)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the network from timm we want to use')
    parser.add_argument('--attack_method', type=str, help = 'Name of the method we are using for finding adversarial examples.')
    parser.add_argument('--indices', type=int, help ='indice where to start the search.')
    parser.add_argument('--batch_size', type=int, help ='indice where to start the search.')
    args = parser.parse_args()
    
    model_name = args.model_name
    attack_method = args.attack_method
    indice =  args.indices
    batch_size = args.batch_size

    run_verification(model_name, attack_method,indice,batch_size)
    