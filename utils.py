import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm


# Returns a Gabor kernel
def gabor(sz, sigma, theta, lambd, gamma, phi):
    radius = (int(sz / 2.0), int(sz / 2.0))
    [x, y] = torch.meshgrid(torch.arange(-radius[0], radius[0] + 1),
                            torch.arange(-radius[1], radius[1] + 1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    left = torch.exp(-(x1 ** 2 + gamma ** 2 * y1 ** 2) / (2 * sigma ** 2))
    right = torch.cos(2 * np.pi * x1 / lambd + phi)
    gab = left * right
    return gab


# Returns the smallest odd integer larger than x
def odd(x) -> int:
    x = np.ceil(x)
    to_add = 1 - x % 2
    return x + to_add


# Returns a Laplacian of Gaussian filter of SD sigma
def LoG(sigma=1.5):
    sz = odd(9 * sigma)
    radius = (int(sz / 2.), int(sz / 2.))
    [x, y] = torch.meshgrid(torch.arange(-radius[0], radius[0] + 1),
                            torch.arange(-radius[1], radius[1] + 1))

    log = -(1 / (2 * np.pi * sigma ** 6)) * \
          (x ** 2 + y ** 2 - 2 *
           (sigma ** 2)) * \
            (torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))

    on_log = f.relu(log)
    off_log = f.relu(-log)
    on_log_sum = on_log.sum()
    off_log_sum = off_log.sum()

    on_log *= off_log_sum / on_log_sum

    return (on_log - off_log) * 0.4519 / log.amax()


# Rearranges and returns already shuffled
# images and labels data in batch_size long batches
def batch_data(batch_size, images, labels):
    image_batches = []
    label_batches = []
    for batch_index in tqdm(range(0, len(images), batch_size), desc=f"Batching data: "):
        image_batches.append(tuple(images[batch_index:batch_index + batch_size]))
        label_batches.append(tuple(labels[batch_index:batch_index + batch_size]))

    return image_batches, label_batches
