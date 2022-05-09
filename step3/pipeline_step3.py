import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.Cityscapes import Cityscapes
from model.build_BiSeNet import BiSeNet


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=4, stride=2, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)

        return x


def make(config):

    # load datasets and dataloaders
    dataset_source = None
    dataloader_source = DataLoader(dataset_source, batch_size=config.batch_size, shuffle=True)

    dataset_target = None
    dataloader_target = DataLoader(dataset_target, batch_size=config.batch_size, shuffle=True)

    # build models
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda

    model_seg = BiSeNet(config.num_classes, config.context_path)
    if torch.cuda.is_available() and config.use_gpu:
        model = torch.nn.DataParallel(model_seg).cuda()

    model_discr = Discriminator(in_channels=config.num_classes)
    if torch.cuda.is_available() and config.use_gpu:
        model_discr = torch.nn.DataParallel(model_discr).cuda()

    # build optimizer
    optim_seg = torch.optim.SGD(model_seg.parameters(), config.learning_rate, momentum=0.9, weight_decay=1e-4)
    optim_discr = torch.optim.Adam(model_discr.parameters(), config.learning_rate, betas=(0.9, 0.99))

    # load loss function
    loss_seg = None
    loss_discr = None

    return model_seg, model_discr, loss_seg, loss_discr, optim_seg, optim_discr, dataloader_source, dataloader_target
