from dataset.Cityscapes import Cityscapes
from torch.utils.data import DataLoader
import torch
from dataset.GTA5 import GTA5
from train_fda.fda_utils import FDA_source_to_target
from utils import denormalize_image, format_image_print, format_label_print, get_legend_handles
import matplotlib.pyplot as plt


# define datasets
batch_size = 4

dataset_src = GTA5('../GTA5', 'train', [512, 1024], False)
dataloader_src = DataLoader(dataset_src, batch_size=batch_size, shuffle=False)

dataset_tgt = Cityscapes('../Cityscapes', 'train', [512, 1024], False)
dataloader_tgt = DataLoader(dataset_tgt, batch_size=batch_size, shuffle=False)

fig, axarr = plt.subplots(batch_size, 3, figsize=(10, 10))

# FFT for a batch
img_batch_src, lbl_batch = next(iter(dataloader_src))
img_batch_denorm_src = denormalize_image(
    img_batch_src, dataset_tgt.mean, dataset_tgt.std)

img_batch_tgt, _ = next(iter(dataloader_tgt))
img_batch_denorm_tgt = denormalize_image(
    img_batch_tgt, dataset_tgt.mean, dataset_tgt.std)

img_batch_src2tgt = FDA_source_to_target(
    img_batch_denorm_src, img_batch_denorm_tgt, beta=0.01)

patches = get_legend_handles(dataset_tgt.labels, dataset_tgt.palette)


# show results
for idx in range(batch_size):
    img_src = img_batch_denorm_src[idx]
    img_src2tgt = img_batch_src2tgt[idx]
    lbl = lbl_batch[idx]

    axarr[idx, 0].imshow(format_image_print(img_src))
    axarr[idx, 1].imshow(format_image_print(img_src2tgt))
    axarr[idx, 2].imshow(format_label_print(lbl, dataset_tgt.palette))

    plt.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.)

plt.show()
