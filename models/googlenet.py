import torch
import torch.nn as tnn
from nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout, Linear, Flatten

__all__ = [
    'GoogLeNet'
]


class BasicConv2d(tnn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv2d = Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm2d(out_channels, eps=0.001)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(tnn.Module):

    def __init__(
        self, 
        in_channels,
        num_1x1,
        num_3x3_reduce,
        num_3x3,
        num_5x5_reduce,
        num_5x5,
        pool_proj
    ):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, num_1x1, kernel_size=1)
        self.branch2 = tnn.Sequential(
            BasicConv2d(in_channels, num_3x3_reduce, kernel_size=1),
            BasicConv2d(num_3x3_reduce, num_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = tnn.Sequential(
            BasicConv2d(in_channels, num_5x5_reduce, kernel_size=1),
            BasicConv2d(num_5x5_reduce, num_5x5, kernel_size=5, padding=3)
        )
        self.branch4 = tnn.Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(tnn.Module):
    """The architecture is based on the paper: Going deeper with convolutions.
    (https://arxiv.org/abs/1409.4842)
    """

    def __init__(self, num_classes=1000, dropout=0.2):
        super().__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = MaxPool2d(kernel_size=3, stride=2)

        self.conv2a = BasicConv2d(64, 64, kernel_size=1)
        self.conv2b = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = MaxPool2d(kernel_size=3, stride=2)

        self.inception3a = Inception(
            in_channels=192, 
            num_1x1=64, 
            num_3x3_reduce=96, 
            num_3x3=128, 
            num_5x5_reduce=16, 
            num_5x5=32, 
            pool_proj=32)
        self.inception3b = Inception(
            in_channels=256, 
            num_1x1=128, 
            num_3x3_reduce=128, 
            num_3x3=192, 
            num_5x5_reduce=32, 
            num_5x5=96, 
            pool_proj=64)
        self.maxpool3 = MaxPool2d(kernel_size=3, stride=2)

        self.inception4a = Inception(
            in_channels=480, 
            num_1x1=192, 
            num_3x3_reduce=96, 
            num_3x3=208, 
            num_5x5_reduce=16, 
            num_5x5=48, 
            pool_proj=64)
        self.inception4b = Inception(
            in_channels=512, 
            num_1x1=160, 
            num_3x3_reduce=112, 
            num_3x3=224, 
            num_5x5_reduce=24, 
            num_5x5=64, 
            pool_proj=64)
        self.inception4c = Inception(
            in_channels=512, 
            num_1x1=128, 
            num_3x3_reduce=128, 
            num_3x3=256, 
            num_5x5_reduce=24, 
            num_5x5=64, 
            pool_proj=64)
        self.inception4d = Inception(
            in_channels=512, 
            num_1x1=112, 
            num_3x3_reduce=144, 
            num_3x3=288, 
            num_5x5_reduce=32, 
            num_5x5=64, 
            pool_proj=64)
        self.inception4e = Inception(
            in_channels=528, 
            num_1x1=256, 
            num_3x3_reduce=160, 
            num_3x3=320, 
            num_5x5_reduce=32, 
            num_5x5=128, 
            pool_proj=128)
        self.maxpool4 = MaxPool2d(kernel_size=2, stride=2)

        self.inception5a = Inception(
            in_channels=832, 
            num_1x1=256, 
            num_3x3_reduce=160, 
            num_3x3=320, 
            num_5x5_reduce=32, 
            num_5x5=128, 
            pool_proj=128)
        self.inception5b = Inception(
            in_channels=832, 
            num_1x1=384, 
            num_3x3_reduce=192, 
            num_3x3=384, 
            num_5x5_reduce=48, 
            num_5x5=128, 
            pool_proj=128)

        self.avgpool = tnn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.dropout = Dropout(p=dropout)
        self.fc = Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x