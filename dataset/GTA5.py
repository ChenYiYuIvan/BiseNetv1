import os
from dataset.GeneralDataset import GeneralDataset


class GTA5(GeneralDataset):
    def __init__(self, path, mode, img_size, data_augmentation):
        super().__init__(path, mode, img_size, data_augmentation)

        # store paths of images
        with open(os.path.join(self.path, 'train.txt'), 'r') as f:
            image_names = f.readlines()
            self.image_path_list = [os.path.join(self.path, 'images', self.get_path(image_name)) for image_name in
                                    image_names]
            self.label_path_list = [os.path.join(self.path, 'labels', self.get_path(image_name)) for image_name in
                                    image_names]

    @staticmethod
    def get_path(image_name):
        return image_name.rstrip()
