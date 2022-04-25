import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Cityscapes(Dataset):
    def __init__(self, path, mode, img_size):
        super().__init__()
        self.path = path
        self.mode = mode
        self.img_size = img_size

        with open(os.path.join(self.path, f'{mode}.txt'), 'r') as f:
            image_names = f.readlines()
            self.image_path_list = [os.path.join(self.path, 'images', self.get_image_path(image_name)) for image_name in
                                    image_names]
            self.label_path_list = [os.path.join(self.path, 'labels', self.get_label_path(image_name)) for image_name in
                                    image_names]

        self.image_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.PILToTensor(),
        ])

    def get_image_path(self, image_name):
        return image_name.split('/')[1].rstrip()

    def get_label_path(self, image_name):
        return image_name.split('/')[1].rstrip().replace('leftImg8bit', 'gtFine_labelIds')

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        label_path = self.label_path_list[idx]
        image = self.image_transform(Image.open(image_path)).float()
        label = self.image_transform(Image.open(label_path))
        return image, label
