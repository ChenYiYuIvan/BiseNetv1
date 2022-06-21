import json
import torch
from dataset.Cityscapes import Cityscapes
from dataset.GTA5 import GTA5
from dataset.Cityscapes import Cityscapes
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


batch_size = 4
num_classes = 19

dataset_cityscapes_train = Cityscapes(
    '../Cityscapes', 'train', [512, 1024], False)
dataloader_cityscapes_train = DataLoader(
    dataset_cityscapes_train, batch_size=batch_size, shuffle=False)
freq_cityscapes_train = torch.zeros(20, dtype=int)

for (data, label) in dataloader_cityscapes_train:
    for type in range(num_classes):
        freq_cityscapes_train[type] = torch.sum(label == type)

    # unlabeled
    freq_cityscapes_train[19] = torch.sum(label == 255)


dataset_cityscapes_val = Cityscapes('../Cityscapes', 'val', [512, 1024], False)
dataloader_cityscapes_val = DataLoader(
    dataset_cityscapes_val, batch_size=batch_size, shuffle=False)
freq_cityscapes_val = torch.zeros(20, dtype=torch.int)

for (data, label) in dataloader_cityscapes_val:
    for type in range(num_classes):
        freq_cityscapes_val[type] = torch.sum(label == type)

    # unlabeled
    freq_cityscapes_val[19] = torch.sum(label == 255)


dataset_gta = GTA5('../GTA5', 'train', [512, 1024], False)
dataloader_gta = DataLoader(dataset_gta, batch_size=batch_size, shuffle=False)

freq_gta = torch.zeros(20, dtype=int)

for (data, label) in dataloader_gta:
    for type in range(num_classes):
        freq_gta[type] = torch.sum(label == type)

    # unlabeled
    freq_gta[19] = torch.sum(label == 255)

freq_cityscapes_train = (freq_cityscapes_train / torch.sum(freq_cityscapes_train)).numpy()
freq_cityscapes_val = (freq_cityscapes_val / torch.sum(freq_cityscapes_val)).numpy()
freq_gta = (freq_gta / torch.sum(freq_gta)).numpy()

with open('./dataset/info.json', 'r') as f:
    data = json.load(f)
    labels = data['label']

x_axis = np.arange(len(labels))

plt.figure(figsize=(20,4))

plt.bar(x_axis - 0.2, freq_cityscapes_train, width=0.2, color='b', align='center', label='Cityscapes Train')
plt.bar(x_axis, freq_cityscapes_val, width=0.2, color='g', align='center', label='Cityscapes Val')
plt.bar(x_axis + 0.2, freq_gta, width=0.2, color='r', align='center', label='GTA5')

plt.xticks(x_axis, labels, size = 'medium')
plt.ylabel('Frequency', size  = 'large')
plt.legend(prop={'size': 'x-large'})
plt.show()