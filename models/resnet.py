import torch.nn as tnn
import torch.nn.functional as F

from nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, ReLU

__all__ = [
    'ResNet'
]


def conv3x3(in_planes, planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


class Block(tnn.Module):
    """A building block for ResNet"""

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)

        self.relu = ReLU() # inplace=True
        self.downsample = downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x

    
class ResNet(tnn.Module):
    """The architecture is based on the paper: Deep Residual Learning for Image Recognition.
    (https://arxiv.org/abs/1512.03385)
    """

    def __init__(self, blocks, num_classes=1000):
        super().__init__()

        self.conv1 = Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU() # inplace=True
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(64, 64, blocks[0])
        self.conv3_x = self._make_layer(64, 128, blocks[1], stride=2)
        self.conv4_x = self._make_layer(128, 256, blocks[2], stride=2)
        self.conv5_x = self._make_layer(256, 512, blocks[3], stride=2)

        self.avgpool = tnn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or in_planes != planes:
            downsample = tnn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                BatchNorm2d(planes)
            )
            
        layers = []
        layers.append(Block(in_planes, planes, stride=stride, downsample=downsample))

        for _ in range(1, blocks):
            layers.append(Block(planes, planes))

        return tnn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x