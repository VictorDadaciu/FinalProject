import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import utils
import torchvision.models as models

# Default values for sigma and lambda
s_tuple = (1.5, 2)
l_tuple = (4, 7)


def build_gabor_weights(sig, lam, gamma):
    # Generate simple neuron gabor filters
    sigmas = []
    thetas = []
    lambdas = []
    gammas = []
    phis = []
    # 2 * 8 * 4 * 4 * 1 = 256
    for sigma in np.around(np.linspace(sig[0], sig[1], num=2), 3):
        for theta in np.arange(0, np.pi, np.pi / 8):
            for lambd in np.around(np.linspace(lam[0], lam[1], num=4), 3):
                for phi in np.arange(0, np.pi, np.pi / 4):
                    thetas.append(theta)
                    sigmas.append(sigma)
                    lambdas.append(lambd)
                    gammas.append(gamma)
                    phis.append(phi)
    return np.array(sigmas), np.array(thetas), np.array(lambdas), \
           np.array(gammas), np.array(phis)


# Gabor filter bank
class Gabor(nn.Module):
    def __init__(self, ksize, filters, channels, stride=1):
        super().__init__()
        self.filters = torch.zeros((filters, channels, ksize, ksize))
        self.ksize = ksize
        self.padding = ksize // 2
        self.channels = channels
        self.stride = stride

    def build(self, sigma, theta, lambd, gamma, psi):
        active_channel = torch.randint(0, self.channels, (256,))
        for i in range(len(self.filters)):
            self.filters[i, active_channel[i]] = utils.gabor(self.ksize,
                                                             sigma[i],
                                                             theta[i],
                                                             lambd[i],
                                                             gamma[i],
                                                             psi[i])
        self.filters = nn.Parameter(self.filters, requires_grad=False)

    def forward(self, x):
        return f.conv2d(x, self.filters, None,
                        stride=self.stride, padding=self.padding)


# The complete V1 layer
class V1_LNP(nn.Module):
    def __init__(self, channels, sig, lam):
        super().__init__()
        sigma, theta, lambd, gamma, phi = build_gabor_weights(sig, lam, 0.7)
        self.gfb_1 = Gabor(11, 256, channels)
        self.gfb_2 = Gabor(11, 256, channels)
        self.gfb_1.build(sigma, theta, lambd, gamma, phi)
        self.gfb_2.build(sigma, theta, lambd, gamma, phi + np.pi / 2)

    def forward(self, x):
        # Linear layer
        conv_1 = self.gfb_1(x)
        conv_2 = self.gfb_2(x)

        # Nonlinear layer
        simple = f.relu(conv_1)
        complx = torch.sqrt(conv_1 ** 2 + conv_2 ** 2) / np.sqrt(2)

        # Poisson layer
        filtered = torch.cat((simple, complx), 1)
        eps = 10e-5
        filtered += torch.distributions.normal.Normal(torch.zeros_like(filtered),
                                                      scale=1).rsample() * \
                    torch.sqrt(f.relu(filtered.clone()) + eps)
        return filtered


# LNPNet
class VOneMobileNet(nn.Module):
    def __init__(self, num_classes, channels=3, sig=None, lam=None):
        super().__init__()
        # If sigma and lambda parameters are left unchanged, choose defaults
        if sig is None:
            sig = s_tuple
        if lam is None:
            lam = l_tuple
        self.v1 = V1_LNP(channels, sig, lam)

        # Bottlenekc layer convolves from 512 channels down to 3
        self.bottleneck = nn.Conv2d(512, 3, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(self.bottleneck.weight,
                                mode="fan_out", nonlinearity="relu")
        self.mobile_net = models.mobilenet_v2(**{"num_classes": num_classes})

    def forward(self, x):
        x = self.v1(x)
        x = self.bottleneck(x)
        return self.mobile_net(x)
