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

---
## BiseNet: real-time semantic segmentation network

Code to train the BiseNet network on a subset of Cityscapes dataset to perform semantic segmentation:

```
!python train_segmentation/main_segmentation.py
```

Loss used: $L_{seg}(I_{s})$ taken from this [paper](https://arxiv.org/abs/1808.00897)

---
## Unsupervised domain adaptation

Code to perform semantic segmentation with unsupervised domain adaptation via adversarial learning on GTA5 as source dataset and Cityscapes as target dataset:

```
!python train_adversarial/main_adversarial.py
```

Loss used:
- segmentation network: $L(I_{s}, I_{t}) = L_{seg}(I_{s}) + \lambda_{adv}L_{adv}(I_{t})$
- discriminator: $L_{d}(P)$

($\lambda_{adv} = 0.001$, while $L_{adv}(I_{t})$ and $L_{d}(P)$ are taken from this [paper](https://arxiv.org/abs/1802.10349))

---
## FDA: Fourier domain adaptation

Code to apply Fast Fourier Transform on GTA5 dataset to improve the performance of unsupervised domain adaptation methods

```
!python train_fda/main_fda.py
```

Loss used:
- FDA only: $L(I_{s\rightarrow t}) = L_{seg}(I_{s\rightarrow t}) + \lambda_{ent}L_{ent}(I_{t})$
- FDA + adversarial:
  - segmentation network: $L(I_{s\rightarrow t}, I_{t}) = L_{seg}(I_{s\rightarrow t}) + \lambda_{ent}L_{ent}(I_{t}) + \lambda_{adv}L_{adv}(I_{t})$
  - discriminator: $L_{d}(P)$

($\lambda_{ent} = 0.005$, while $L_{ent}(I_{t})$ is taken from this [paper](https://arxiv.org/abs/2004.05498))

---
## MBT: Multi-band transfer

Code to evaluate performance of averaging predictions of 3 different models trained with FDA with different betas:

```
!python train_mbt_sst/val_mbt.py
```

Betas used: $\beta_{1} = 0.01$, $\beta_{2} = 0.05$, $\beta_{3} = 0.09$

---
## Self-supervised learning

Code to produce pseudolabels for Cityscapes dataset with MBT using the best model for each beta, which are then used to perform self-supervised learning on models using FDA using the same 3 betas, which are in turn used to perform a second round of MBT:

```
!python train_mbt_sst/generate_pseudolabels.py
!python train_mbt_sst/main_fda_sst.py
!python train_mbt_sst/val_mbt.py
```

Loss used:
- FDA only + MBT + SST: $L_{sst}(I_{s\rightarrow t}, I_{t-pseudo}) = L_{seg}(I_{s\rightarrow t}) + \lambda_{ent}L_{ent}(I_{t}) + L_{ce}(I_{t-pseudo})$
- FDA + adversarial + MBT + SST:
  - segmentation network: $L_{sst}(I_{s\rightarrow t}, I_{t-pseudo}, I_{t}) = L_{seg}(I_{s\rightarrow t}) + \lambda_{ent}L_{ent}(I_{t}) + L_{ce}(I_{t-pseudo}) + \lambda_{adv}L_{adv}(I_{t})$
  - discriminator: $L_{d}(P)$

($L_{ce}(I_{t-pseudo})$ is taken from this [paper](https://arxiv.org/abs/2004.05498))

---
## Acknowledgment

Part of the code is adapted from the following projects:
- [Starting code](https://github.com/taveraantonio/BiseNetv1)
- [Unsupervised domain adaptation](https://github.com/wasidennis/AdaptSegNet)
- [FDA and self-supervised learning](https://github.com/YanchaoYang/FDA)
