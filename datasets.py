import torch
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms, datasets
import numpy as np

mnist = None
fmnist = None
cifar10 = None
cifar100 = None
gtsrb = None
tiny_image_net = None


class Dataset:
    def __init__(self, name, classes, channels, train, test, epochs, step_size):
        self.name = name + f" epochs={epochs}"
        self.classes = classes
        self.channels = channels
        self.train = train
        self.test = test
        self.epochs = epochs
        self.step_size = step_size


def MNIST(batch_size=100, epochs=3, step_size=3):
    global mnist
    if mnist is not None:
        return mnist

    train = datasets.MNIST("./Data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("./Data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    train_set = torch.utils.data.DataLoader(train,
            batch_size=batch_size, shuffle=True)
    test_set = torch.utils.data.DataLoader(test,
            batch_size=batch_size, shuffle=True)

    mnist = Dataset("MNIST", 10, 1, train_set, test_set, epochs, step_size)
    return mnist


def FashionMNIST(batch_size=100, epochs=5, step_size=5):
    global fmnist
    if fmnist is not None:
        return fmnist

    train = datasets.FashionMNIST("./Data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.FashionMNIST("./Data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    train_set = torch.utils.data.DataLoader(train,
            batch_size=batch_size,
            shuffle=True)
    test_set = torch.utils.data.DataLoader(test,
            batch_size=batch_size,
            shuffle=True)
    fmnist = Dataset("FashionMNIST", 10, 1, train_set, test_set, epochs, step_size)
    return fmnist


def CIFAR10(batch_size=100, epochs=30, step_size=12):
    global cifar10
    if cifar10 is not None:
        return cifar10

    train = datasets.CIFAR10("./Data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.CIFAR10("./Data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    train_set = torch.utils.data.DataLoader(train,
            batch_size=batch_size,
            shuffle=True)
    test_set = torch.utils.data.DataLoader(test,
            batch_size=batch_size,
            shuffle=True)

    cifar10 = Dataset("CIFAR10", 10, 3, train_set, test_set, epochs, step_size)
    return cifar10


def CIFAR100(batch_size=100, epochs=40, step_size=15):
    global cifar100
    if cifar100 is not None:
        return cifar100

    train = datasets.CIFAR100("./Data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.CIFAR100("./Data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    train_set = torch.utils.data.DataLoader(train,
            batch_size=batch_size,
            shuffle=True)
    test_set = torch.utils.data.DataLoader(test,
            batch_size=batch_size,
            shuffle=True)

    cifar100 = Dataset("CIFAR100", 100, 3, train_set, test_set, epochs, step_size)
    return cifar100


def GTSRB(batch_size=50, epochs=10, step_size=10):
    global gtsrb
    if gtsrb is not None:
        return gtsrb

    print("Loading GTSRB dataset...")

    data_sets = {}
    path = "Data/GTSRB"
    # For each type of set load in its corresponding data
    for data_type in ["train", "test"]:
        data_sets[data_type] = []
        images = np.load(f"{path}/{data_type}_images.npy",
                            allow_pickle=True).tolist()
        labels = np.load(f"{path}/{data_type}_labels.npy",
                         allow_pickle=True).tolist()
        # Batch data into batch_size batches
        for batch_index in tqdm(range(0, len(images), batch_size),
                                desc=f"Loading {data_type}ing data: "):
            data_sets[data_type].append((
                torch.FloatTensor(images[batch_index:batch_index + batch_size]),
                torch.LongTensor(labels[batch_index:batch_index + batch_size])))
    gtsrb = Dataset("GTSRB", 43, 3,
                    data_sets["train"],
                    data_sets["test"],
                    epochs, step_size)
    print("Done!")
    return gtsrb


def TinyImageNet(has_to_train=True, batch_size=50, epochs=50, step_size=20):
    global tiny_image_net
    if tiny_image_net is not None:
        return tiny_image_net

    print("Loading TinyImageNet dataset...")

    data_sets = {}
    path = "Data/TinyImageNet"
    # For each type of set load in its corresponding mega batches of data
    data_types = {"train": 0, "test": 1}
    # Load training data only if has_to_train is True
    # to prevent unnecessary lengthy loading
    if has_to_train:
        data_types["train"] = 4

    for data_type in data_types:
        data_sets[data_type] = []
        # Batch data into batch_size batches
        for batch in range(data_types[data_type]):
            images = np.load(f"{path}/{data_type}_images_{batch}.npy",
                             allow_pickle=True).tolist()
            labels = np.load(f"{path}/{data_type}_labels_{batch}.npy",
                             allow_pickle=True).tolist()
            for batch_index in tqdm(range(0, len(images), batch_size),
                                    desc=f"Loading {data_type}ing data: "):
                data_sets[data_type].append((
                    torch.FloatTensor(images[batch_index:batch_index + batch_size]),
                    torch.LongTensor(labels[batch_index:batch_index + batch_size])))
    tiny_image_net = Dataset("TinyImageNet", 200, 3,
                             data_sets["train"],
                             data_sets["test"], epochs, step_size)
    print("Done!")
    return tiny_image_net
