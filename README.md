# A Hybrid Paradigm Integrate Self-attention and Convolution on 3D Medical Image Segmentation

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

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







## A Hybrid Paradigm Integrate Self-attention and Convolution on 3D Medical Image Segmentation

**Paper: [CoTr: Efficient 3D Medical Image Segmentation
by bridging CNN and Transformer](https://arxiv.org/pdf/2103.03024.pdf
).** 

Meta AI Research, FAIR

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick

[Paper] [Project] [Demo] [Dataset] [Blog] [BibTeX]

## Requirements
CUDA 11.0<br />
Python 3.7<br /> 
Pytorch 1.7<br />
Torchvision 0.8.2<br />

## Usage

### 0. Installation
* Install Pytorch1.7, nnUNet and CoTr as below
  
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

cd nnUNet
pip install -e .

cd CoTr_package
pip install -e .
```

### 1. Data Preparation
* Download [BCV dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
* Preprocess the BCV dataset according to the uploaded nnUNet package.
* Training and Testing ID are in `data/splits_final.pkl`.

### 2. Training 
cd CoTr_package/CoTr/run

* Run `nohup python run_training.py -gpu='0' -outpath='CoTr' 2>&1 &` for training.

### 3. Testing 
* Run `nohup python run_training.py -gpu='0' -outpath='CoTr' -val --val_folder='validation_output' 2>&1 &` for validation.

### 4. Citation
If this code is helpful for your study, please cite:

```
@article{xie2021cotr,
  title={CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation},
  author={Xie, Yutong and Zhang, Jianpeng and Shen, Chunhua and Xia, Yong},
  booktitle={MICCAI},
  year={2021}
}
  
```

### 5. Acknowledgements
Part of codes are reused from the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Thanks to Fabian Isensee for the codes of nnU-Net.

### Contact
Yutong Xie (xuyongxie@mail.nwpu.edu.cn)
