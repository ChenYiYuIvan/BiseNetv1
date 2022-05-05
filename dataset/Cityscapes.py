import os
from dataset.GeneralDataset import GeneralDataset


class Cityscapes(GeneralDataset):
    def __init__(self, path, mode, img_size):
        super().__init__(path, img_size)
        self.mode = mode

        # store paths of images
        with open(os.path.join(self.path, f'{mode}.txt'), 'r') as f:
            image_names = f.readlines()
            self.image_path_list = [os.path.join(self.path, 'images', self.get_image_path(image_name)) for image_name in
                                    image_names]
            self.label_path_list = [os.path.join(self.path, 'labels', self.get_label_path(image_name)) for image_name in
                                    image_names]

    @staticmethod
    def get_image_path(image_name):
        return image_name.split('/')[1].rstrip()

    @staticmethod
    def get_label_path(image_name):
        return image_name.split('/')[1].rstrip().replace('leftImg8bit', 'gtFine_labelIds')
