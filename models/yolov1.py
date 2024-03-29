import torch.nn as tnn

from nn import BatchNorm2d, Conv2d, Flatten, LeakyReLU, Linear, MaxPool2d

__all__ = [
    'DarkNetV1'
]


class Block(tnn.Module):
    """A building block for DarkNet"""

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1, normalization=True, activation="leaky"):
        super().__init__()

        bias = False if normalization else True 

        layers = []
        layers.append(Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        if normalization:
            layers.append(BatchNorm2d(planes))
        if activation == "leaky":
            layers.append(LeakyReLU(0.1, inplace=True))
        
        self.conv = tnn.Sequential(*layers)

    def forward(self, x):
        y = self.conv(x)
        return y


class DarkNetV1(tnn.Module):
    """The architecture is based on the paper: You Only Look Once: Unified, Real-Time Object Detection
    (https://arxiv.org/abs/1506.02640)

    Shape
        (N, 3, 448, 448) -> (N, 5*B + C, 7, 7)
        where N is the batch size, B is the number of bounding boxes, and C is the number of classes.
    """

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov1-tiny.cfg
    def __init__(self, num_bboxes=2, num_classes=20):
        super().__init__()
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.block1 = Block(3, 16)
        self.maxpool1 = MaxPool2d(2, 2)

        self.block2 = Block(16, 32)
        self.maxpool2 = MaxPool2d(2, 2)

        self.block3 = Block(32, 64)
        self.maxpool3 = MaxPool2d(2, 2)

        self.block4 = Block(64, 128)
        self.maxpool4 = MaxPool2d(2, 2)

        self.block5 = Block(128, 256)
        self.maxpool5 = MaxPool2d(2, 2)

        self.block6 = Block(256, 512)
        self.maxpool6 = MaxPool2d(2, 2)

        self.block7 = Block(512, 1024)
        self.block8 = Block(1024, 256)

        self.flatten = Flatten()
        self.fc = Linear(256 * 7 * 7, 7 * 7 * (self.num_bboxes*5 + self.num_classes)) # 1470 = 7 * 7 * (2*5 + 20)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool1(x)

        x = self.block2(x)
        x = self.maxpool2(x)

        x = self.block3(x)
        x = self.maxpool3(x)

        x = self.block4(x)
        x = self.maxpool4(x)

        x = self.block5(x)
        x = self.maxpool5(x)

        x = self.block6(x)
        x = self.maxpool6(x)

        x = self.block7(x)
        x = self.block8(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(-1, 5 * self.num_bboxes + self.num_classes, 7, 7)

        return x