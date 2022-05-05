from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
import numpy as np


# TODO questions to ask:
#   working in range [0, 255] instead of [0,1] is a problem?
#   correct mean and std values for normalization? mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
#   batch normalization? already implemented
#   package for visualization of training (for report) (eg. neptune, weights&biases, tensorboard...)? last 2
#   bisenet for segmentation network? yes
#   is discriminator network architecture pretrained? no
#   sgd or adam optimizer for training? sgd for seg, adam for discr
#   single or multi-level adversarial learning? if multi, how many? single
#   values of parameters in loss functions? nel paper

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
            transforms.PILToTensor(),
            transforms.Lambda(lambda tensor: tensor.float()),
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
