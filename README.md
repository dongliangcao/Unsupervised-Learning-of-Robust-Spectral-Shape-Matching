# pytorch-framework
PyTorch common framework to accelerate network implementation, training and validation.

This framework is inspired by works from [MMLab](https://github.com/open-mmlab), which modularize the data, network,
loss, metric, etc. to make the framework to be flexible, easy to modify and to extend.


## How to use
```bash
# install necessary libs
pip install -r requirements.txt
```
The framework contains six different subfolders:
- networks: all networks should be implemented under the networks folder with {NAME}_network.py filename. 
- datasets: all datasets should be implemented under the datasets folder with {NAME}_dataset.py filename.
- losses: all losses should be implemented under the losses folder with {NAME}_loss.py filename.
- metrics: all metrics should be implemented under the metrics folder with {NAME}_metric.py filename.
- models: all models should be implemented under the models folder with {NAME}_model.py filename.
- utils: all util functions should be implemented under the utils folder with {NAME}_util.py filename.

The training and validation procedure can be defined in the specified .yaml file.
```bash
# training 
CUDA_VISIBLE_DEVICES=gpu_ids python train.py --opt options/train.yaml

# validation/test
CUDA_VISIBLE_DEVICES=gpu_ids python test.py --opt options/test.yaml
```

In the .yaml file for training, you can define all the things related to training such as the experiment name, model, 
dataset, network, loss, optimizer, metrics and other hyper-parameters. Here is an example to train VGG16 for image classification:
```bash
# general setting
name: vgg_train
backend: dp # DataParallel
type: ClassifierModel
num_gpu: auto

# path to resume network
path:
  resume_state: ~

# datasets
datasets:
  train_dataset:
    name: TrainDataset
    type: ImageNet
    data_root: ../data/train_data
  val_dataset:
    name: ValDataset
    type: ImageNet
    data_root: ../data/val_data
  # setting for train dataset
  batch_size: 8

# network setting
networks:
  classifier:
    type: VGG16
    num_classes: 1000

# training setting
train:
  total_iter: 10000
  optims:
    classifier:
      type: Adam
      lr: 1.0e-4
  schedulers:
    classifier:
      type: none
  losses:
    ce_loss:
      type: CrossEntropyLoss

# validation setting
val:
  val_freq: 10000

# log setting
logger:
  print_freq: 100
  save_checkpoint_freq: 10000
```
In the .yaml file for validation, you can define all the things related to validation such as: model, dataset, metrics.
Here is an example:
```bash
# general setting
name: test
backend: dp # DataParallel
type: ClassifierModel
num_gpu: auto
manual_seed: 1234

# path
path:
  resume_state: experiments/train/models/final.pth
  resume: false

# datasets
datasets:
  val_dataset:
    name: ValDataset
    type: ImageNet
    data_root: ../data/test_data

# network setting
networks:
  classifier:
    type: VGG
    num_classes: 1000

# validation setting
val:
  metrics:
    accuracy:
      type: calculate_accuracy
```

## Framework Details
The core of the framework is the **BaseModel** in the [base_model.py](models/base_model.py). The **BaseModel** controls 
the whole training/validation procedure from initialization over training/validation iteration to results saving.
- Initialization:
In the model initialization, it will read the configuration in the .yaml file and construct the 
corresponding networks, datasets, losses, optimizers, metrics, etc. 
- Training/Validation:
In the training/validation procedure, you can refer the training process in the [train.py](./train.py) and the validation process in the [test.py](./test.py).
- Results saving:
The model will automatically save the state_dict for networks, optimizers and other hyperparameters during the training.

The configuration of the framework is down by **Register** in the [registry.py](utils/registry.py). The **Register**
 has a object map (key-value pair). The key is the name of the object, the value is the class of the object. 
There are total 4 different registers for networks, datasets, losses and metrics.
Here is an example to register a new network:
```bash
import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY

@NETWORK_REGISTRY.register()
class MyNet(nn.Module):
  ...
```
