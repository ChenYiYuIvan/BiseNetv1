from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
import numpy as np


# TODO:
#   wandb or tensorboard for visualization
#   sgd for segmentation, adam for discriminator
#   single level adversarial learning
#   values of parameters in loss functions in the paper

class GeneralDataset(Dataset):
    def __init__(self, path, img_size):
        super().__init__()
        self.path = path
        self.img_size = img_size

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
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        self.label_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.Lambda(lambda image: np.asarray(image)),
            transforms.Lambda(lambda pixel: self.label_mapping[pixel]),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor.squeeze(0)),
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        label_path = self.label_path_list[idx]
        image = self.image_transform(Image.open(image_path))
        label = self.label_transform(Image.open(label_path))
        return image, label
