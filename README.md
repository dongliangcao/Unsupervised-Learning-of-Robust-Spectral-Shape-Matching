## [Unsupervised Learning of Robust Spectral Shape Matching (SIGGRAPH/TOG 2023)](https://dongliangcao.github.io/urssm/)
![img](figures/teaser.jpg)

## Installation
```bash 
conda create -n fmnet python=3.8 # create new viertual environment
conda activate fmnet
conda install pytorch cudatoolkit -c pytorch # install pytorch
pip install -r requirements.txt # install other necessary libraries via pip
```

## Dataset
To train and test datasets used in this paper, please download the datasets from the this [link](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link) and put all datasets under ../data/
```Shell
├── data
    ├── FAUST_r
    ├── FAUST_a
    ├── SCAPE_r
    ├── SCAPE_a
    ├── SHREC19_r
    ├── TOPKIDS
    ├── SMAL_r
    ├── DT4D_r
    ├── SHREC20
    ├── SHREC16
    ├── SHREC16_test
```
We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the original authors.

## Data preparation
For data preprocessing, we provide *[preprocess.py](preprocess.py)* to compute all things we need.
Here is an example for FAUST_r.
```python
python preprocess.py --data_root ../data/FAUST_r/ --no_normalize --n_eig 200
```

## Train
To train the model on a specified dataset.
```python
python train.py --opt options/train/faust.yaml 
```
You can visualize the training process in tensorboard.
```bash
tensorboard --logdir experiments/
```

## Test
To test the model on a specified dataset.
```python
python test.py --opt options/test/faust.yaml 
```
The qualitative and quantitative results will be saved in [results](results) folder.

## Texture Transfer
An example of texture transfer is provided in *[texture_transfer.py](texture_transfer.py)*
```python
python texture_transfer.py
```

## Pretrained models
You can find all pre-trained models in [checkpoints](checkpoints) for reproducibility.

## Partial Shape Matching on SHREC’16
There were two issues with the partial shape matching experiments on the SHREC'16 dataset related to the training/test splits [Bracha et al. 2023, Ehm et al. 2024]. Below, we provide additional evaluations that substantiate our claims:
| Geo err (x100)    | CUTS on CUTS | CUTS on HOLES | HOLES on CUTS | HOLES on HOLES | CUTS on CUTS'24* | HOLES on CUTS'24* |
| ----------------  | :----------: | :------------:|:------------: |:------------:  | :---------------:|:-----------------:|
| Ours original**   |   3.3        | 13.7          |5.2            |  9.1           | 3.4              |5.5                |
| Ours new***       |   3.2        | 13.5          |5.6            |  8.2           | 3.2              |5.9                |

\*   CUTS'24 refers to the new test split from [Ehm et al. 2024], the split can be found [here](https://github.com/vikiehm/geometrically-consistent-partial-partial-shape-matching/tree/main/CUTS24).

**  Pretrained on TOSCA

*** Pretrained on FAUST + SCAPE + SMAL + DT4D-H, 25 test-time adaptation iterations

[[Bracha et al. 2023] A. Bracha, T. Dages, R. Kimmel, On Partial Shape Correspondence and Functional Maps, arXiv 2023](https://arxiv.org/abs/2310.14692).

[[Ehm et al. 2024] V. Ehm, M. Gao, P. Roetzer, M. Eisenberger, D. Cremers, F. Bernard, Partial-to-Partial Shape Matching with Geometric Consistency, CVPR 2024](https://arxiv.org/abs/2404.12209).


## Acknowledgement
The implementation of DiffusionNet is based on [the official implementation](https://github.com/nmwsharp/diffusion-net).

The framework implementation is adapted from [Unsupervised Deep Multi Shape Matching](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching).
