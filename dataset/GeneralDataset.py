import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
import numpy as np
import random
from torchvision.transforms.functional import InterpolationMode
from utils import RandomCrop


class GeneralDataset(Dataset):
    def __init__(self, path, mode, img_size, data_augmentation):
        super().__init__()
        self.path = path
        self.mode = mode
        self.img_size = img_size
        self.data_augmentation = data_augmentation

        # attributes defined in children classes
        self.image_path_list = None
        self.label_path_list = None

        # store mapping of labels
        with open('./dataset/info.json', 'r') as f:
            data = json.load(f)
            mappings = data['label2train']
            self.label_mapping = np.array(mappings)[:, 1]
            self.labels = data['label']
            self.palette = np.reshape(data['palette'], 60).tolist()
            self.mean = data['mean']
            self.std = data['std']

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),  # transform into torch tensor with range [0,1]
            transforms.Normalize(self.mean, self.std),  # normalize using ImageNet mean and std
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        label_path = self.label_path_list[idx]
        image = Image.open(image_path)
        label = Image.open(label_path)

        # data augmentation only during training
        augment = self.data_augmentation and self.mode == 'train'

        if not augment:  # need to resize when there is no crop into correct size
            resize = transforms.Resize(self.img_size)
            image = resize(image)
            label = resize(label)

        if augment:  # apply random scale and random crop
            seed = random.random()

            image = self.random_scale(image, interpolation=InterpolationMode.BILINEAR, seed=seed)
            label = self.random_scale(label, interpolation=InterpolationMode.NEAREST, seed=seed)

            image = RandomCrop(self.img_size, seed, pad_if_needed=True)(image)
            label = RandomCrop(self.img_size, seed, pad_if_needed=True)(label)

        image = self.image_transform(image)  # transform into tensor and normalize

        label = np.array(label)  # apply correct labels and transform into tensor
        label = self.label_mapping[label]
        label = torch.from_numpy(label)

        if augment:  # apply random horizontal flip
            flip_bool = random.random() < 0.5

            if flip_bool:
                image = torch.flip(image, [-1])
                label = torch.flip(label, [-1])

        return image, label.squeeze()

    def random_scale(self, tensor, interpolation, seed):
        random.seed(seed)

        scales = [0.75, 1.0, 1.5, 1.75, 2.0]
        scale = random.choice(scales)
        size = (int(self.img_size[0] * scale), int(self.img_size[1] * scale))
        resize = transforms.Resize(size, interpolation)

        return resize(tensor)
