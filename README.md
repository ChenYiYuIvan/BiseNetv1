# Real-time domain adaptation in semantic segmentation

## Setting python environment

```
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/BiSeNet
%env PYTHONPATH=/content/drive/MLDL/Progetto/BiseNetv1

!pip install wandb
import wandb
wandb.login()
```

## BiseNet: real-time semantic segmentation network

Code to train the BiseNet network on a subset of Cityscapes dataset to perform semantic segmentation:

```
!python train_segmentation/main_segmentation.py
```

## Unsupervised domain adaptation

Code to perform semantic segmentation with unsupervised domain adaptation via adversarial learning on GTA5 as source dataset and Cityscapes as target dataset:

```
!python train_adversarial/main_adversarial.py
```

## FDA: Fourier domain adaptation

Code to apply Fast Fourier Transform on GTA5 dataset to improve the performance of unsupervised domain adaptation methods

```
!python train_fda/main_fda.py
```

## MBT: Multi-band transfer

Code to evaluate performance of averaging predictions of 3 different models trained with FDA with different betas:

```
!python train_mbt_sst/val_mbt.py
```

## Self-supervised learning

Code to produce pseudolabels for Cityscapes dataset using MBT, which are then used to perform self-supervised learning on models using FDA using 3 different beta, which are in turn used to perform a second round of MBT:

```
!python train_mbt_sst/generate_pseudolabels.py
!python train_mbt_sst/main_fda_sst.py
!python train_mbt_sst/val_mbt.py
```

## Acknowledgment

Part of the code is adapted from the following projects:
- [Starting code](https://github.com/taveraantonio/BiseNetv1)
- [Unsupervised domain adaptation](https://github.com/wasidennis/AdaptSegNet)
- [FDA and self-supervised learning](https://github.com/YanchaoYang/FDA)
