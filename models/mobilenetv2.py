import torch.nn as tnn
import torch.nn.functional as F

from nn import BatchNorm2d, Flatten, Linear, ReLU6

__all__ = [
    'MobileNetV2'
]

class Bottleneck(tnn.Module):
    """Bottleneck residual block"""

    def __init__(self, in_planes, planes, stride, expansion):
        super().__init__()

        hidden_dim = in_planes * expansion

        layers = []
        if expansion != 1:
            layers.extend([
                # conv1x1, relu6
                tnn.Conv2d(in_planes, hidden_dim, kernel_size=1, bias=False),
                BatchNorm2d(hidden_dim),
                ReLU6()
            ])

        layers.extend([
            # dwise conv3x3, relu6
            tnn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            BatchNorm2d(hidden_dim),
            ReLU6(),
            # conv1x1, linear
            tnn.Conv2d(hidden_dim, planes, kernel_size=1, bias=False),
            BatchNorm2d(planes)
        ])

        self.shortcut = (stride == 1 and in_planes == planes)
        self.bottleneck = tnn.Sequential(*layers)

    def forward(self, x):
        if self.shortcut:
            y = x + self.bottleneck(x)
        else:
            y = self.bottleneck(x)

        return y
                

class MobileNetV2(tnn.Module):
    """The architecture is based on the paper: MobileNetV2: Inverted Residuals and Linear Bottlenecks.
    (https://arxiv.org/abs/1801.04381)

    Shape
        (N, 1, H, W) -> (N, C)
        where N is the batch size, H is the image height, W is the image width, and C is the number of classes.
    """

    def __init__(self, num_classes=1000):
        super().__init__()

        settings = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        in_planes = 1
        planes = 32
        layers = []
        # conv3x3
        layers.extend([
            tnn.Conv2d(in_planes, planes, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            ReLU6()
        ])

        # bottleneck
        in_planes = planes
        for t, c, n, s in settings:
            planes = c
            for i in range(n):
                stride = s if i ==0 else 1
                layers.append(Bottleneck(in_planes, planes, stride, t))
                in_planes = planes

        planes = 1280
        layers.extend([
            # conv1x1
            tnn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(planes),
            ReLU6(),
            # avgpool
            tnn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            Linear(planes, num_classes)
        ])

        self.net = tnn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y