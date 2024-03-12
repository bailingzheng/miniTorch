import torch.nn as tnn

from nn import Conv2d, Dropout, Flatten, Linear, MaxPool2d, ReLU

__all__ = [
    'AlexNet'
]


class AlexNet(tnn.Module):
    """The architecture is based on the paper: ImageNet Classification with Deep Convolutional Neural Networks.
    (https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

    """

    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.net = tnn.Sequential(
            # conv1
            Conv2d(3, 96, kernel_size=11, stride=4, padding=2), # (3, 224, 224) -> (96, 55, 55)
            ReLU(),
            # maxpool1
            MaxPool2d(kernel_size=3, stride=2), # (96, 27, 27)
            # conv2
            Conv2d(96, 256, kernel_size=5, padding=2), # (256, 27, 27)
            ReLU(),
            # maxpool2
            MaxPool2d(kernel_size=3, stride=2), # (256, 13, 13)
            # conv3
            Conv2d(256, 384, kernel_size=3, padding=1), # (384, 13, 13)
            ReLU(),
            # conv4
            Conv2d(384, 384, kernel_size=3, padding=1), # (384, 13, 13)
            ReLU(),
            # conv5
            Conv2d(384, 256, kernel_size=3, padding=1), # (256, 13, 13)
            ReLU(),
            # maxpool3
            MaxPool2d(kernel_size=3, stride=2), # (256, 6, 6)
            Flatten(),
            Dropout(p=dropout),
            # fc1
            Linear(256 * 6 * 6, 4096),
            ReLU(),
            Dropout(p=dropout),
            # fc2
            Linear(4096, 4096),
            ReLU(),
            # fc3
            Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        return x
