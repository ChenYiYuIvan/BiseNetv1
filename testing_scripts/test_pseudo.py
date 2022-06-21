import json
import matplotlib.pyplot as plt
from PIL import Image
import torch
from utils import format_label_print
import numpy as np
from torchvision import transforms as T


gt_path = '../Cityscapes/labels/aachen_000001_000019_gtFine_labelIds.png'
pseudo_path = gt_path.replace('labels', 'pseudo')

gt_lbl = Image.open(gt_path)
pseudo_lbl = Image.open(pseudo_path)

with open('./dataset/info.json', 'r') as f:
    data = json.load(f)
    mappings = data['label2train']
    label_mapping = np.array(mappings)[:, 1]
    palette = np.reshape(data['palette'], 60).tolist()

gt_lbl = np.array(gt_lbl)  # apply correct labels and transform into tensor
gt_lbl = label_mapping[gt_lbl]
gt_lbl = torch.from_numpy(gt_lbl).squeeze(0)
pseudo_lbl = T.Compose([T.PILToTensor()])(pseudo_lbl).squeeze(0)

fig, axarr = plt.subplots(1, 2)

axarr[0].imshow(format_label_print(gt_lbl, palette))
axarr[0].set_title('ground truth', fontsize=35)
axarr[0].axis('off')
axarr[1].imshow(format_label_print(pseudo_lbl, palette))
axarr[1].set_title('pseudo label', fontsize=35)
axarr[1].axis('off')

fig.tight_layout()
plt.show()
