import torch
import torch.nn as tnn
from nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout, Linear, Flatten

__all__ = [
    'InceptionResnetV1'
]


class BasicConv2d(tnn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm2d(out_channels, eps=0.001)
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BlockA(tnn.Module):
    """35x35 grid module"""

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale
        self.branch1 = BasicConv2d(256, 32, kernel_size=1)
        self.branch2 = tnn.Sequential(
            BasicConv2d(256, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )
        self.branch3 = tnn.Sequential(
            BasicConv2d(256, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )
        self.conv = Conv2d(96, 256, kernel_size=1)
        self.relu = ReLU()

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y = torch.cat((y1, y2, y3), 1)
        y = self.conv(y)
        y = y*self.scale + x
        y = self.relu(y)
        return y


class BlockB(tnn.Module):
    """17x17 grid module"""

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale
        self.branch1 = BasicConv2d(896, 128, kernel_size=1)
        self.branch2 = tnn.Sequential(
            BasicConv2d(896, 128, kernel_size=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), padding=(3, 0))
        )
        self.conv = Conv2d(256, 896, kernel_size=1)
        self.relu = ReLU()

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y = torch.cat((y1, y2), 1)
        y = self.conv(y)
        y = y*self.scale + x
        y = self.relu(y)
        return y


class BlockC(tnn.Module):
    """8x8 grid module"""

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale
        self.branch1 = BasicConv2d(1792, 192, kernel_size=1)
        self.branch2 = tnn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), padding=(1, 0))
        )
        self.conv = Conv2d(384, 1792, kernel_size=1)
        self.relu = ReLU()

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y = torch.cat((y1, y2), 1)
        y = self.conv(y)
        y = y*self.scale + x
        y = self.relu(y)
        return y


class ReductionA(tnn.Module):
    """35x35 to 17x17 reduction module"""

    def __init__(self):
        super().__init__()

        self.branch1 = MaxPool2d(kernel_size=3, stride=2)
        self.branch2 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        self.branch3 = tnn.Sequential(
            BasicConv2d(256, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        return torch.cat((y1, y2, y3), 1)


class ReductionB(tnn.Module):
    """17x17 to 8x8 reduction module"""

    def __init__(self):
        super().__init__()

        self.branch1 = MaxPool2d(kernel_size=3, stride=2)
        self.branch2 = tnn.Sequential(
            BasicConv2d(896, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        self.branch3 = tnn.Sequential(
            BasicConv2d(896, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        self.branch4 = tnn.Sequential(
            BasicConv2d(896, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat((y1, y2, y3, y4), 1)


class InceptionResnetV1(tnn.Module):
    """The architecture is based on the paper: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning.
    (https://arxiv.org/abs/1602.07261)
    """

    def __init__(self, num_classes=1000, dropout=0.2):
        super().__init__()

        # 149x149x32
        self.conv1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        # 147x147x32
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)
        # 147x147x64
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        # 73x73x64
        self.maxpool = MaxPool2d(kernel_size=3, stride=2)
        # 73x73x80
        self.conv4 = BasicConv2d(64, 80, kernel_size=1)
        # 71x71x192
        self.conv5 = BasicConv2d(80, 192, kernel_size=3)
        # 35x35x256
        self.conv6 = BasicConv2d(192, 256, kernel_size=3, stride=2)
        # 35x35x256
        self.repeat1 = tnn.Sequential(
            BlockA(scale=0.17),
            BlockA(scale=0.17),
            BlockA(scale=0.17),
            BlockA(scale=0.17),
            BlockA(scale=0.17)
        )
        # 17x17x896
        self.reduction1 = ReductionA()
        # 17x17x896
        self.repeat2 = tnn.Sequential(
            BlockB(scale=0.10),
            BlockB(scale=0.10),
            BlockB(scale=0.10),
            BlockB(scale=0.10),
            BlockB(scale=0.10),
            BlockB(scale=0.10),
            BlockB(scale=0.10),
            BlockB(scale=0.10),
            BlockB(scale=0.10),
            BlockB(scale=0.10)
        )
        # 8x8x1792
        self.reduction2 = ReductionB()
        # 8x8x1792
        self.repeat3 = tnn.Sequential(
            BlockC(scale=0.20),
            BlockC(scale=0.20),
            BlockC(scale=0.20),
            BlockC(scale=0.20),
            BlockC(scale=0.20)
        )
        # 1x1x1792
        self.avgpool = tnn.AdaptiveAvgPool2d((1,1))
        self.flatten = Flatten()
        self.dropout = Dropout(dropout)
        self.fc = Linear(1792, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.repeat1(x)
        x = self.reduction1(x)
        x = self.repeat2(x)
        x = self.reduction2(x)
        x = self.repeat3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x