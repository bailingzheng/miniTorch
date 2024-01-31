import torch
import torch.nn as tnn

from nn import BatchNorm2d, Conv2d, MaxPool2d, ReLU

__all__ = [
    'UNet'
]

def conv3x3(in_planes, planes):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)


def upconv(in_planes, planes):
    """2x2 transposed convolution"""
    return tnn.ConvTranspose2d(in_planes, planes, kernel_size=2, stride=2)


class Block(tnn.Module):
    """A building block for UNet"""

    def __init__(self, in_planes, planes):

        super().__init__()

        self.conv1 = conv3x3(in_planes, planes)
        self.bn1 = BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)

        self.relu = ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UNet(tnn.Module):
    """The architecutre is based on the paper: U-Net: Convolutional Networks for Biomedical Image Segmentation.
    (https://arxiv.org/abs/1505.04597)

    Shape
        (N, 3, H, W) -> (N, C, H, W)
        where N is the batch size, H is the image height, W is the image width, and C is the number of classes.

    """

    def __init__(self, planes=32, num_classes=23):
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

        self.conv2 = Conv2d(planes, num_classes, kernel_size=1)

    def forward(self, x):
        y1 = self.block1(x)

        m1 = self.maxpool1(y1)
        y2 = self.block2(m1)

        m2 = self.maxpool2(y2)
        y3 = self.block3(m2)

        m3 = self.maxpool3(y3)
        y4 = self.block4(m3)

        m4 = self.maxpool4(y4)
        y5 = self.block5(m4)

        u1 = self.upconv1(y5)
        y6 = self.block6(torch.cat((u1, y4), dim=1))

        u2 = self.upconv2(y6)
        y7 = self.block7(torch.cat((u2, y3), dim=1))

        u3 = self.upconv3(y7)
        y8 = self.block8(torch.cat((u3, y2), dim=1))

        u4 = self.upconv4(y8)
        y9 = self.block9(torch.cat((u4, y1), dim=1))

        y = self.conv2(y9)

        return y