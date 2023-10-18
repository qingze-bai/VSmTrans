# A Hybrid Paradigm Integrate Self-attention and Convolution on 3D Medical Image Segmentation

[[`Paper`](#)] [[`Dataset`](https://amos22.grand-challenge.org/)] [[`BibTeX`](#)]

![Variable-Shape design](assets/fig01.jpg?raw=true)


The **VSmTrans** is a hybrid Transformer that tightly integrates self-attention and convolution into one paradigm, which enjoy the benefits of a large receptive field and strong inductive bias from both sides.


## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install nnUNet:

```
cd nnUNet
pip install -e .
```

Install VSmTrans:

```
cd VSmTrans_package
pip install -e .
```

## Getting Started

cd VSmTrans_package/vsmt/run

```
python run_training.py  --network_trainer="vsmtTrainerV2_AMOS" --task=1 --fold="all" --outpath='vsmt'
```

## Citation


