import torch.nn as tnn

from nn import Conv2d, Dropout, Flatten, Linear, MaxPool2d, ReLU

__all__ = [
    'VGG',
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19'
]

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend([Conv2d(in_channels, v, kernel_size=3, padding=1), ReLU(inplace=True)])
            in_channels = v
    return tnn.Sequential(*layers)

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(tnn.Module):
    """The architecture is based on the paper: Very Deep Convolutional Networks for Large-Scale Image Recognition. 
    (https://arxiv.org/abs/1409.1556)
    """

    def __init__(self, layers, num_classes=1000, dropout=0.5):
        super().__init__()
        self.layers = layers
        self.avgpool = tnn.AdaptiveAvgPool2d((7, 7))
        self.flatten = Flatten()
        self.classifier = tnn.Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(p=dropout),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Dropout(p=dropout),
            Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, Conv2d):
                tnn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                tnn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                tnn.init.normal_(m.weight, mean=0, std=0.01)
                tnn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def vgg11():
    model = VGG(make_layers(cfgs["A"]))
    return model

def vgg13():
    model = VGG(make_layers(cfgs["B"]))
    return model

def vgg16():
    model = VGG(make_layers(cfgs["D"]))
    return model

def vgg19():
    model = VGG(make_layers(cfgs["E"]))
    return model