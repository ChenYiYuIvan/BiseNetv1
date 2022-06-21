from dataset.Cityscapes import Cityscapes
from torch.utils.data import DataLoader
from dataset.GTA5 import GTA5
from train_fda.fda_utils import FDA_source_to_target
from utils import denormalize_image, format_image_print
import matplotlib.pyplot as plt


# define datasets
batch_size = 1

dataset_src = GTA5('../GTA5', 'train', [512, 1024], False)
dataloader_src = DataLoader(dataset_src, batch_size=batch_size, shuffle=False)

dataset_tgt = Cityscapes('../Cityscapes', 'train', [512, 1024], False)
dataloader_tgt = DataLoader(dataset_tgt, batch_size=batch_size, shuffle=False)

fig, axarr = plt.subplots(2, 3)

# FFT for a batch
img_batch_src, lbl_batch = next(iter(dataloader_src))
img_src = img_batch_src[0]

img_denorm_src = denormalize_image(
    img_batch_src, dataset_tgt.mean, dataset_tgt.std)

img_batch_tgt, _ = next(iter(dataloader_tgt))

img_denorm_tgt = denormalize_image(
    img_batch_tgt, dataset_tgt.mean, dataset_tgt.std)

img_src2tgt1 = FDA_source_to_target(
    img_denorm_src, img_denorm_tgt, beta=0.001)[0]

img_src2tgt2 = FDA_source_to_target(
    img_denorm_src, img_denorm_tgt, beta=0.01)[0]
    
img_src2tgt3 = FDA_source_to_target(
    img_denorm_src, img_denorm_tgt, beta=0.10)[0]

img_src2tgt4 = FDA_source_to_target(
    img_denorm_src, img_denorm_tgt, beta=0.25)[0]

img_src2tgt5 = FDA_source_to_target(
    img_denorm_src, img_denorm_tgt, beta=0.50)[0]

axarr[0, 0].imshow(format_image_print(img_denorm_src[0]))
axarr[0, 0].set_title('source image')
axarr[0, 0].axis('off')
axarr[0, 1].imshow(format_image_print(img_src2tgt1))
axarr[0, 1].set_title(r'$ \beta = 0.001 $')
axarr[0, 1].axis('off')
axarr[0, 2].imshow(format_image_print(img_src2tgt2))
axarr[0, 2].set_title(r'$ \beta = 0.01 $')
axarr[0, 2].axis('off')
axarr[1, 0].imshow(format_image_print(img_src2tgt3))
axarr[1, 0].set_title(r'$ \beta = 0.10 $')
axarr[1, 0].axis('off')
axarr[1, 1].imshow(format_image_print(img_src2tgt4))
axarr[1, 1].set_title(r'$ \beta = 0.25 $')
axarr[1, 1].axis('off')
axarr[1, 2].imshow(format_image_print(img_src2tgt5))
axarr[1, 2].set_title(r'$ \beta = 0.50 $')
axarr[1, 2].axis('off')

fig.tight_layout()

plt.show()