import os
from dataset.GeneralDataset import GeneralDataset
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import random
from torchvision.transforms.functional import InterpolationMode
from utils import RandomCrop


class CityscapesPseudo(GeneralDataset):
    def __init__(self, path, mode, img_size, data_augmentation):
        super().__init__(path, mode, img_size, data_augmentation)

        if self.mode == 'val':
            raise ValueError('CityscapesPseudo is only for generating pseudolabels for self-supervised training')

        # store paths of images
        with open(os.path.join(self.path, 'train.txt'), 'r') as f:
            image_names = f.readlines()
            self.image_path_list = [os.path.join(self.path, 'images', self.get_image_path(image_name)) for image_name in
                                    image_names]
            self.label_path_list = [os.path.join(self.path, 'labels', self.get_label_path(image_name)) for image_name in
                                    image_names]
            self.pseudo_path_list = [os.path.join(self.path, 'pseudo', self.get_label_path(image_name)) for image_name in
                                    image_names]

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        label_path = self.label_path_list[idx]
        pseudo_path = self.pseudo_path_list[idx]
        image = Image.open(image_path)

        if self.mode == 'generate':  # in this case the label is not used
            label = Image.open(label_path)
        elif self.mode == 'train':
            label = Image.open(pseudo_path)
        else:
            raise ValueError('CityscapesPseudo is only for generating pseudolabels and for self-supervised training')

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

        image_to_tensor = transforms.Compose([transforms.PILToTensor()])
        label = image_to_tensor(label)

        if augment:  # apply random horizontal flip
            flip_bool = random.random() < 0.5

            if flip_bool:
                image = torch.flip(image, [-1])
                label = torch.flip(label, [-1])

        return image, label.squeeze(), image_path, pseudo_path

    @staticmethod
    def get_image_path(image_name):
        return image_name.split('/')[1].rstrip()

    @staticmethod
    def get_label_path(image_name):
        return image_name.split('/')[1].rstrip().replace('leftImg8bit', 'gtFine_labelIds')
