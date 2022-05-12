from torch import nn


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


class DepthwiseSeparableDiscriminator(nn.Module):
    def __init__(self, in_ch, out_ch=1, kernel_size=4, stride=2, padding=1):
        super().__init__()

        self.depth1 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.point1 = nn.Conv2d(in_ch, 64, kernel_size=kernel_size, stride=stride, padding=padding)

        self.depth2 = nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding, groups=64)
        self.point2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)

        self.depth3 = nn.Conv2d(128, 128, kernel_size=kernel_size, stride=stride, padding=padding, groups=128)
        self.point3 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding)

        self.depth4 = nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, padding=padding, groups=256)
        self.point4 = nn.Conv2d(256, 512, kernel_size=kernel_size, stride=stride, padding=padding)

        self.depth5 = nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding, groups=512)
        self.point5 = nn.Conv2d(512, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.depth1(x)
        x = self.point1(x)
        x = self.leaky_relu(x)

        x = self.depth2(x)
        x = self.point2(x)
        x = self.leaky_relu(x)

        x = self.depth3(x)
        x = self.point3(x)
        x = self.leaky_relu(x)

        x = self.depth4(x)
        x = self.point4(x)
        x = self.leaky_relu(x)

        x = self.depth5(x)
        x = self.point5(x)

        return x

