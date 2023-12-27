# A Hybrid Paradigm Integrate Self-attention and Convolution on 3D Medical Image Segmentation

[[`Paper`](#)] [[`Dataset`](https://amos22.grand-challenge.org/)] [[`BibTeX`](#)]

![Variable-Shape design](assets/fig01.jpg?raw=true)


The **VSmTrans** is a hybrid Transformer that tightly integrates self-attention and convolution into one paradigm, which enjoy the benefits of a large receptive field and strong inductive bias from both sides.

## Installation

The code requires `python>=3.9`, as well as `pytorch>1.12`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install nnUNet:
```
cd nnUNet
pip install -e .
```

Our model relies on the nnU-Net framework, and its needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to set a few environment variables. Please follow the instructions

```
export nnUNet_raw="raw data path"
export nnUNet_preprocessed="preprocessed data path"
export nnUNet_results="result path"
```

## Dataset Format
Datasets must be located in the nnUNet_raw folder (which you either define when installing nnU-Net or export/set every time you intend to run nnU-Net commands!). Each segmentation dataset is stored as a separate 'Dataset'. Datasets are associated with a dataset ID, a three digit integer, and a dataset name (which you can freely choose): For example, Dataset005_Prostate has 'Prostate' as dataset name and the dataset id is 5. Datasets are stored in the nnUNet_raw folder like this:
```
nnUNet_raw/
├── Dataset001_BrainTumour
├── Dataset002_Heart
├── Dataset003_Liver
├── Dataset004_Hippocampus
├── Dataset005_Prostate
├── ...
```
Within each dataset folder, the following structure is expected:
```
Dataset001_BrainTumour/
├── dataset.json
├── imagesTr
├── imagesTs  # optional
└── labelsTr
```
Json format details please refer to [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). 

**Note that the naming of each data is required to specify the input modal type**, e.g. BraTS. This dataset hat four input channels: FLAIR (0000), T1w (0001), T1gd (0002) and T2w (0003).
```
nnUNet_raw/Dataset001_BrainTumour/
├── dataset.json
├── imagesTr
│   ├── BRATS_001_0000.nii.gz
│   ├── BRATS_001_0001.nii.gz
│   ├── BRATS_001_0002.nii.gz
│   ├── BRATS_001_0003.nii.gz
│   ├── BRATS_002_0000.nii.gz
│   ├── BRATS_002_0001.nii.gz
│   ├── BRATS_002_0002.nii.gz
│   ├── BRATS_002_0003.nii.gz
│   ├── ...
├── imagesTs
│   ├── BRATS_485_0000.nii.gz
│   ├── BRATS_485_0001.nii.gz
│   ├── BRATS_485_0002.nii.gz
│   ├── BRATS_485_0003.nii.gz
│   ├── BRATS_486_0000.nii.gz
│   ├── BRATS_486_0001.nii.gz
│   ├── BRATS_486_0002.nii.gz
│   ├── BRATS_486_0003.nii.gz
│   ├── ...
└── labelsTr
    ├── BRATS_001.nii.gz
    ├── BRATS_002.nii.gz
    ├── ...
```
Here is another example of the second dataset of the MSD, which has only one input channel:
```
nnUNet_raw/Dataset002_Heart/
├── dataset.json
├── imagesTr
│   ├── la_003_0000.nii.gz
│   ├── la_004_0000.nii.gz
│   ├── ...
├── imagesTs
│   ├── la_001_0000.nii.gz
│   ├── la_002_0000.nii.gz
│   ├── ...
└── labelsTr
    ├── la_003.nii.gz
    ├── la_004.nii.gz
    ├── ...
```
Remember: For each training case, all images must have the same geometry to ensure that their pixel arrays are aligned. Also make sure that all your data is co-registered!




self.network = VSmixTUnet(
    in_channels=1,
    out_channels=16,
    feature_size=24,
    split_size=[1, 3, 5, 7],
    window_size=6,
    num_heads=[3, 6, 12, 24],
    img_size=[96, 96, 96],
    depths=[2, 2, 2, 2],
    patch_size=(2, 2, 2),
    do_ds=True
)

## Experiment planning and preprocessing
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -c 3d_fullres
```
Where DATASET_ID is the dataset id (duh). it is recommended that you use the --verify_dataset_integrity command. This will check for some of the most common error sources! For more information about all the options available to you please run nnUNetv2_plan_and_preprocess -h.

## Training
```
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [additional options, see -h]
```
DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of the 5-fold-cross-validation is trained. More details please refer [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

## Inference
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c 3d_fullres -f FOLD [--save_probabilities]
```

Note that per default, inference will be done with all 5 folds from the cross-validation as an ensemble. We very strongly recommend you use all 5 folds. Thus, all 5 folds must have been trained prior to running inference.

If you wish to make predictions with a single model, train the all fold and specify it in `nnUNetv2_predict` with `-f all`

You can also run it directly, for the default configuration see [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework.

## Model
Our model exists in the directory: `nnUNet/nnunetv2/training/nnUNetTrainer/VSmTrans.py`. You can easily to use it in own framework as follows:
```
self.network = VSmixTUnet(
    in_channels=1,
    out_channels=16,
    feature_size=48,
    split_size=[1, 3, 5, 7],
    window_size=6,
    num_heads=[3, 6, 12, 24],
    img_size=[96, 96, 96],
    depths=[2, 2, 2, 2],
    patch_size=(2, 2, 2),
    do_ds=True
)
```

## Citation