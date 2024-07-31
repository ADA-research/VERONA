# Robustness Experiment Box

This VERONA package simplifies your experiment pipeline for performing local robustness verification on your networks and dataset. 
The entire package is class-based, which means that extending the existing configurations is accessible and easy. 
With only one script it should be possible to run an entire experiment with various networks, images and epsilons. 
The package can be used to create robustness distributions [Bosman, Hoos and van Rijn, 2023] and per-class robustness distributions [Bosman et al., 2024] 

If you have any suggestions to enhance this package, feel free to create an issue or a pull-request.
## Setup
- clone the repository locally
- create new environment ```conda create -n verona python=3.10```
- activate the environment ```conda activate verona```
- install dependencies ```pip install -r requirements.txt```
- install package locally (editable install for development) ```pip install -e .```

This package was tested only on python version 3.10 and we cannot guarentee it working on any other python version at this point

## Experiment Folder
The following structure for an experiment folder is currently supported:

```
experiment/
|-- data/
|   |-- labels.csv
|   |-- images/
|       |-- mnist_0.npy
|       |-- mnist_1.npy
|       |-- ...
|-- networks/
|   |-- mnist-net_256x2.onnx
|   |-- mnist-net_256x4.onnx
|   |-- ...
```

The images can be placed in it optionally, if one wants to execute the experiments using custom data. Otherwise, Pytorch Datasets can be used too and no image folder has to be created.

## Results folder

The following structure for an result folder is currently supported:
```
experiment_results/
|-- results/
|   |-- results_df.csv
|   |-- boxplot.png
|   |-- ecdf_plot.png
|   |-- hist_figure.png
|   |-- kde_plot.png
|-- slurm/
|   |-- slurm_.out
|   |-- config_.txt
|   |-- ...
|-- tmp/
|   |-- mnist-net_256x2\
|   |   |-- image_0\
|   |   |  |-- epsilons_df.csv
|   |   |  |-- property___.vnnlib
|   |   |  |-- property___.vnnlib.compiled
|   |   |  |-- ...
|   |   |-- ...
|   |-- ...
|-- yaml/
|   |--verification_context____.yaml
|   |-- ...
|-- configuration.json
```

The ```yaml``` and ```slurm``` folders only exist when working with the multiple jobs example. 

The ```results_df.csv``` is structured as follows at this point. 

```
network_path | image_id | tmp_path | epsilon_value | smallest_sat_value |total_time
```


## Available Verifiers

### AutoVerify
Auto verifiy offers the following verifiers:
- nnenum https://github.com/stanleybak/nnenum
- AB-Crown https://github.com/Verified-Intelligence/alpha-beta-CROWN
- VeriNet https://github.com/vas-group-imperial/VeriNet
- Oval-Bab https://github.com/oval-group/oval-bab

If the auto verify module is used, the verifiers have to be installed as described in the auto-verify documentation.

https://github.com/ADA-research/auto-verify


### Custom verifiers
Custom verifiers can be implemented to, by using the VerificationModule interface.

## Testing
Core parts of the package can be tested using pytest. In addition, the package was tested using various datasets described below.

### Pytest
To execute the pytests, first install the package with the dev dependencies using  ```pip install '.[dev]'```.
Then, the tests can be executed using ```pytest tests```

### Datasets
- The package was tested on the MNIST and the CIFAR10 dataset. Example scripts for executing the package on mnist or a custom dataset can be found in the ```scripts``` folder
## Tutorial 
### Example Scripts
There are a few example scripts that can be found in the ```scripts``` folder. All these examples are made to be run out of the ```scripts``` folder, otherwise the paths might break. 
- ```create_robustness_distribution_from_test_dataset.py``` gives an example of how to run VERONA with custom dataset and ab-crown. 
- ```create_robustness_dist_on_pytorch_dataset.py``` shows how to load in a pytorch dataset and run VERONA with one-to-any verification on one-to-one verification where a target is specified.
- The folder ```multiple_jobs``` shows an example of how to create a job for each network-image pair and run them on different nodes via SLURM. This is especially useful when working on a computer cluster.
  This also makes it possible to distribute tasks over CPU and GPU for different verifiers in the same experiment for example.

All verifiers can run on both GPU and CPU, except for ab-crown, which can only be employed on the GPU. So the resources necessary to run a script depend on the verifier used.

### Tutorial Notebook
In addition, in the notebooks folder a jupyter notebook is provided to give an overview about the components in the package and how to use them.

## Related Papers
This package was created to simplify reproducing and extending the results of two different lines of work of the ADA research group. Please consider citing these works when using this package for your research. 
### Robustness distributions 
- @article{BosEtAl23,
    author = "Bosman, Annelot W. and Hoos, Holger H. and van Rijn, Jan N.",
    title = "A Preliminary Study of Critical Robustness Distributions in Neural Network Verification",
    year = "2023",
    journal = "6th Workshop on Formal Methods for ML-Enabled Autonomous Systems (FoMLAS) co-located with the 35th International Conference on Computer Aided Verification (CAV 2023)",
    booktitle = "to appear"
  
- https://ada.liacs.leidenuniv.nl/papers/BosEtAl23.pdf
- An extended version of this work is currently under review. 
### Per-class robustness distributions
- @inproceedings{BosEtAl24,
    author = {Bosman, Annelot W. and M\“unz, Anna L. and Hoos, Holger H. and van Rijn, Jan N.},
    title = {{A Preliminary Study to Examining Per-Class Performance Bias via Robustness Distributions}},
    year = {2024},
    booktitle = {The 7th International Symposium on AI Verification (SAIV) co-located with the 36th International Conference on Computer Aided Verification (CAV 2024)}
}
- https://ada.liacs.leidenuniv.nl/papers/BosEtAl24.pdf

