import torch.nn as tnn

from nn import BatchNorm2d, Conv2d, LeakyReLU, ReLU, Tanh

__all__ = [
    'Generator',
    'Discriminator'
]


class Generator(tnn.Module):
    """A network that maps the latent space vector z to a RGB image. 
    The architecture is based on the paper: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.
    (http://arxiv.org/abs/1511.06434)

    Parameters
        ngpu - Number of GPUs available. Use 0 for CPU mode.
        nz - Size of z latent vector.
        ngf - Size of feature maps in generator.
        nc - Number of channels in the training images.

    Shape
        (N, nz) -> (N, nc, 64, 64)
    """

    def __init__(self, ngpu, nz=100, ngf=64, nc=3):
        super().__init__()
        self.ngpu = ngpu
        self.net = tnn.Sequential(
            # conv1
            tnn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False), # (nz, 1, 1) -> (ngf * 8, 4, 4)
            BatchNorm2d(ngf * 8),
            ReLU(inplace=True),
            # conv2
            tnn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False), # (ngf * 4, 8, 8)
            BatchNorm2d(ngf * 4),
            ReLU(inplace=True),
            # conv3
            tnn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False), # (ngf * 2, 16, 16)
            BatchNorm2d(ngf * 2),
            ReLU(inplace=True),
            # conv4
            tnn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False), # (ngf, 32, 32)
            BatchNorm2d(ngf),
            ReLU(inplace=True),
            # conv5
            tnn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=True), # (nc, 64, 64)
            Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(tnn.Module):
    """A binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake). 
    
    Parameters
        ngpu - Number of GPUs available. Use 0 for CPU mode.
        nc - Number of channels in the training images.
        ndf - Size of feature maps in discriminator.

    Shape
        (N, nc, 64, 64) -> (N, 1)
    """

    def __init__(self, ngpu, nc=3, ndf=64):
        super().__init__()
        self.ngpu = ngpu
        self.net = tnn.Sequential(
            # conv1
            Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=True), # (nc, 64, 64) -> (ndf, 32, 32)
            LeakyReLU(negative_slope=0.2, inplace=True),
            # conv2
            Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), # (ndf * 2, 16, 16)
            BatchNorm2d(ndf * 2),
            LeakyReLU(negative_slope=0.2, inplace=True),
            # conv3
            Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), # (ndf * 4, 8, 8)
            BatchNorm2d(ndf * 4),
            LeakyReLU(negative_slope=0.2, inplace=True),
            # conv4
            Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), # (ndf * 8, 4, 4)
            BatchNorm2d(ndf * 8),
            LeakyReLU(negative_slope=0.2, inplace=True),
            # conv5
            Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=True), # (1, )
            tnn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)