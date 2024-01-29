import torch
import torch.nn as tnn

from nn import Conv2d, ReLU, MaxPool2d

__all__ = [
    'UNet'
]

def conv3x3(in_planes, planes):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)


def upconv(in_planes, planes):
    """3x3 transposed convolution"""
    return tnn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=2)


class Block(tnn.Module):
    """A building block for UNet"""

    def __init__(self, in_planes, planes):

        super().__init__()

        self.conv1 = conv3x3(in_planes, planes)
        self.bn1 = tnn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = tnn.BatchNorm2d(planes)

        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.relu(self.bn2(self.conv2(x)))

        return x


class UNet(tnn.Module):
    """The architecutre is based on the paper: U-Net: Convolutional Networks for Biomedical Image Segmentation.
    (https://arxiv.org/abs/1505.04597)

    """

    def __init__(self, planes = 32, num_classes=23):
        super().__init__()

        self.block1 = Block(3, planes)

        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)
        self.block2 = Block(planes, planes * 2)

        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2)
        self.block3 = Block(planes * 2, planes * 4)

        self.maxpool3 = MaxPool2d(kernel_size=2, stride=2)
        self.block4 = Block(planes * 4, planes * 8)

        self.maxpool4 = MaxPool2d(kernel_size=2, stride=2)
        self.block5 = Block(planes * 8, planes * 16)

        self.upconv1 = upconv(planes * 16, planes * 8)
        self.block6 = Block(planes * 16, planes * 8)

        self.upconv2 = upconv(planes * 8, planes * 4)
        self.block7 = Block(planes * 8, planes * 4)

        self.upconv3 = upconv(planes * 4, planes * 2)
        self.block8 = Block(planes * 4, planes * 2)

        self.upconv4 = upconv(planes * 2, planes)
        self.block9 = Block(planes * 2, planes)

        self.conv1 = Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = tnn.BatchNorm2d(planes)
        self.relu = ReLU()

        self.conv2 = Conv2d(planes, num_classes, kernel_size=1)

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(self.maxpool1(y1))
        y3 = self.block3(self.maxpool2(y2))
        y4 = self.block4(self.maxpool3(y3))
        y5 = self.block5(self.maxpool4(y4))

        y4 = self.block6(torch.cat((self.upconv1(y5), y4), dim=1))
        y3 = self.block7(torch.cat((self.upconv2(y4), y3), dim=1))
        y2 = self.block8(torch.cat((self.upconv3(y3), y2), dim=1))
        y1 = self.block9(torch.cat((self.upconv4(y2), y1), dim=1))

        y = self.relu(self.bn(self.conv1(y1)))
        y = self.conv2(y)

        return y