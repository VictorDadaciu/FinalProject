import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import utils
import torchvision.models as models
import torchvision.models.mobilenet as mn

kernels = 32


def build_gabor_weights(ksize, sig, lam, psi, channels):
    on_sum_filters = torch.zeros((kernels, channels, ksize, ksize))
    off_sum_filters = torch.zeros((kernels, channels, ksize, ksize))
    on_max_filters = torch.zeros((kernels, 1, ksize, ksize))
    off_max_filters = torch.zeros((kernels, 1, ksize, ksize))

    index = 0
    gamma = 0.71
    zero = torch.zeros(ksize, ksize)
    for theta in np.arange(0, np.pi, np.pi / 8):
        for phi in np.arange(0, np.pi, np.pi / 4):
            # Base gabor filter
            kern = utils.gabor(ksize, sig, theta, lam, gamma, phi)

            # Separate filters by positive and
            # negative values: ON - positive; OFF - negative
            on_kern = torch.maximum(zero, kern)
            off_kern = torch.maximum(zero, -kern)

            # Calculate normalising values
            total = on_kern.sum() + off_kern.sum()
            maxi = max(torch.amax(on_kern), torch.amax(off_kern))

            # Append weights to their respective lists
            for i in range(channels):
                on_sum_filters[index, i] = on_kern * psi / total
                off_sum_filters[index, i] = off_kern * psi / total
            on_max_filters[index, 0] = on_kern * psi / maxi
            off_max_filters[index, 0] = off_kern * psi / maxi

            index += 1

    return on_sum_filters, off_sum_filters, on_max_filters, off_max_filters


# Converts RGB encoded to LAB format by first
# converting from RGB to XYZ and then from XYZ to LAB
def RGB2LAB(rgb):
    grtr = rgb.clone()
    grtr[grtr < 0.0405] = 0
    grtr = ((grtr + 0.055) / 1.055) ** 2.4
    grtr *= 100

    less = rgb.clone()
    less[less > 0.0405] = 0
    less = less / 12.92
    less *= 100

    rgb = grtr.add(less)
    xyz = torch.zeros_like(rgb)

    xyz[:, 0] = rgb[:, 0] * 0.4124 + rgb[:, 1] * 0.3576 + rgb[:, 2] * 0.1805
    xyz[:, 1] = rgb[:, 0] * 0.2126 + rgb[:, 1] * 0.7152 + rgb[:, 2] * 0.0722
    xyz[:, 2] = rgb[:, 0] * 0.0193 + rgb[:, 1] * 0.1192 + rgb[:, 2] * 0.9505
    xyz[:, 0] /= 95.047
    xyz[:, 1] /= 100.000
    xyz[:, 2] /= 108.883

    # Hack that removes nan values after cube rooting the elements
    # greater than 0.008856 of xyz
    # Caused rare crashes while running PGD attacks

    # Both clean and attack accuracy unchanged after
    # introduction of this hack
    grtr = xyz.clone()
    grtr[grtr < 0.008856] = 0
    grtr = (grtr.abs() + 1e-5) ** (1 / 3)
    grtr[torch.isnan(grtr)] = 0

    less = xyz.clone()
    less[less > 0.008856] = 0
    less = 7.787 * less + 16 / 116

    xyz = grtr.add(less)
    lab = torch.zeros_like(xyz)

    lab[:, 0] = 116 * xyz[:, 1] - 16
    lab[:, 1] = 500 * (xyz[:, 0] - xyz[:, 1])
    lab[:, 2] = 200 * (xyz[:, 1] - xyz[:, 2])

    lab[:, 0] = lab[:, 0] / 100
    lab[:, 1] = (lab[:, 1] + 128) / 255
    lab[:, 2] = (lab[:, 2] + 128) / 255

    return lab


# Processes inputs using a Laplacian of Gaussian filter
class Preprocess(nn.Module):
    def __init__(self, sig=1., channels=3):
        super(Preprocess, self).__init__()
        self.filter = utils.LoG(sig).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.channels = channels
        self.padding = self.filter.size()[2] // 2

        self.filter = nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x):
        x = torch.tanh(2 * np.pi * f.conv2d(x,
                                            self.filter,
                                            None,
                                            stride=1,
                                            padding=self.padding,
                                            groups=self.channels))
        return x


# Gabor filter bank
class SumGabor(nn.Module):
    def __init__(self, ksize, filters):
        super(SumGabor, self).__init__()
        self.ksize = ksize
        self.padding = ksize // 2

        self.filters = filters
        self.filters = nn.Parameter(self.filters, requires_grad=False)

    def forward(self, x):
        return f.conv2d(x, self.filters, None, stride=1, padding=self.padding)


# Gabor filter bank
class MaxGabor(nn.Module):
    def __init__(self, ksize, filters):
        super(MaxGabor, self).__init__()
        self.ksize = ksize
        self.padding = ksize // 2

        self.filters = torch.flip(filters, [2, 3])
        self.filters = nn.Parameter(self.filters, requires_grad=False)

    def forward(self, x):
        return f.conv2d(x, self.filters, None, stride=1,
                        padding=self.padding,
                        groups=32)


class V1_PCBC(nn.Module):
    def __init__(self, channels, iterations, sig, lam, sigp):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)

        self.preprocess = Preprocess(sig=sigp, channels=channels)

        # Build gabor filter banks.
        # Max GFBs are normalised by the maximum value across
        # both on and off kernels and will be flipped for true
        # convolution

        # Sum GFBs will be normalised by the sum across
        # both on and off kernels and will NOT be flipped
        # (cross-correlation)
        ksize = 9
        psi = 5000
        on_sum, off_sum, on_max, off_max = \
            build_gabor_weights(ksize, sig, lam, psi, channels)
        self.sum_gabor_on = SumGabor(ksize=ksize, filters=on_sum)
        self.sum_gabor_off = SumGabor(ksize=ksize, filters=off_sum)
        self.max_gabor_on = MaxGabor(ksize=ksize, filters=on_max)
        self.max_gabor_off = MaxGabor(ksize=ksize, filters=off_max)

        self.epsilon_1 = 0.0001
        self.epsilon_2 = 50
        self.t = iterations
        self.channels = channels

    def forward(self, x):
        # Convert to LAB format only if in RGB format
        if self.channels == 3:
            x = RGB2LAB(x)

        # Separate preprocessed output into on and off
        # tensors based on on-center/off-surround and
        # off-center/on-surround neuron functionality
        x = self.preprocess(x)
        x_on = f.relu(x)
        x_off = f.relu(-x)

        # The shape of the prediction neurons (y) is
        # identical to the input in all but one
        # dimension: the channels dimension.
        # There are "kernels" Gabor kernels so the
        # final size of y will be (N, kernels, h, w),
        # where N is the size of the batch, h and w are
        # the resolution of the images in the batch.

        # Initialise a tensor of this shape with zeros
        shape_y = list(x.size())
        shape_y[1] = kernels
        shape_y = tuple(shape_y)
        y = torch.zeros(shape_y)
        y = y.to("cuda:0")

        # Repeat PC/BC algorithm for t iterations
        for _ in range(self.t):
            # Equation 9
            # Separate error into on and off channels, similarly to input
            e_on = x_on / (self.epsilon_2 +
                           self.max_gabor_on(y).sum(1).unsqueeze(1))
            e_off = x_off / (self.epsilon_2 +
                             self.max_gabor_off(y).sum(1).unsqueeze(1))

            # Equation 10
            # Based on error, update prediction neurons
            y = (self.epsilon_1 + y) * \
                (self.sum_gabor_on(e_on) +
                 self.sum_gabor_off(e_off))
        return y


class VOneMobileNet(nn.Module):
    def __init__(self, num_classes, channels=3, iterations=3, sig=1, lam=1.5, sigp=1):
        super().__init__()

        self.v1 = V1_PCBC(channels, iterations, sig, lam, sigp)

        # Bottleneck layer convolves from kernels channels to 16 channels.
        self.bottleneck = nn.Conv2d(kernels, 16,
                                    kernel_size=1,
                                    stride=1,
                                    bias=False)
        nn.init.kaiming_normal_(self.bottleneck.weight,
                                mode="fan_out",
                                nonlinearity="relu")

        # Change first layer of MobileNetV2 to accept 16 channels
        # instead of the usual 3 and the number of classes
        # to the custom value sent through constructor instead
        # of the default 1000
        self.mobile_net = models.mobilenet_v2(**{"num_classes": num_classes})
        self.mobile_net.features[0] = \
            mn.ConvBNReLU(16, 32, stride=2, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        x = self.v1(x)
        x = self.bottleneck(x)
        x = self.mobile_net(x)
        return x
