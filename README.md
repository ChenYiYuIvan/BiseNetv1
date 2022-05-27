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

## Self-supervised learning

Code to produce pseudolabels for Cityscapes dataset with a network trained using FDA, which are then used to perform self-supervised learning with multi-band transfer:

```
!python train_mbt_sst/generate_pseudolabels.py
!python train_mbt_sst/main_fda_sst.py
```

## Acknowledgment

Part of the code is adapted from the following projects:

<ul>
    <li> [Starting code](https://github.com/taveraantonio/BiseNetv1) </li>
    <li> [Unsupervised domain adaptation](https://github.com/wasidennis/AdaptSegNet) </li>
    <li> [FDA and self-supervised learning](https://github.com/YanchaoYang/FDA) </li>
<ul>
