import os
import numpy as np
from tqdm import tqdm
import utils
import torch
import statistics as st

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from foolbox import PyTorchModel, accuracy
from foolbox.attacks import L1PGD, L2PGD, LinfPGD, GaussianBlurAttack, \
    L2ClippingAwareAdditiveGaussianNoiseAttack, \
    L2ClippingAwareAdditiveUniformNoiseAttack, \
    L2ContrastReductionAttack, \
    L2AdditiveUniformNoiseAttack
from foolbox.distances import LpDistance


class Attack:
    def __init__(self, name, attack, epsilons):
        self.name = name
        self.attack = attack
        self.epsilons = epsilons

    def __call__(self, model, images, labels):
        return self.attack(model, images, labels, epsilons=self.epsilons)


def pytorch_model(model):
    return PyTorchModel(model, bounds=(0, 1))


# Prints and returns the base accuracy of a model to a certain test_data.
# If test_data is None, then the benign testing data of the model is used.
def acc(mdl, test_data=None, desc=""):
    model = pytorch_model(mdl.load_for_eval())

    if test_data is None:
        test_data = mdl.data.test
        desc = f"Clean Accuracy"

    # Run through every batch and calculate total accuracy
    # across batches and count number of batches for final
    # percentage.
    batches = 0
    total_acc = None
    for data in tqdm(test_data, desc=f"{desc}: "):
        images, labels = data
        if mdl.repeat:
            images = images.repeat(1, 3, 1, 1)
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")
        batch_acc = accuracy(model, images, labels)
        if total_acc is None:
            total_acc = batch_acc
        else:
            total_acc += batch_acc
        batches += 1

    if desc == f"Clean Accuracy":
        print(mdl.name)
        print(f"{desc}: {(total_acc / batches) * 100:4.1f} %")

    return (total_acc / batches) * 100


# Loads the dashc attack specified in "data", the name of the data set,
# and runs and logs the results of the attacks.
def dashc(mdls, data, batch_size=50):
    # Path where data is found
    attacks_path = f"Data/{data}/{data}"
    # If folder does not exist, do not run folder
    if data != "CIFAR-10-C" and data != "CIFAR-100-C" and data != "Tiny-ImageNet-C":
        print("Not appropriate data name! Must be "
              "CIFAR-10-C, CIFAR-100-C, or Tiny-ImageNet-C.")
        return

    # Separate labels by severity i.e. 10000 elements per severity
    labels_by_severity = []
    total = 10000
    all_labels = np.load(f"{attacks_path}/labels.npy")
    for i in range(0, len(all_labels), total):
        labels_by_severity.append(all_labels[i:i + total])

    # Pass through all .npy images files and load them
    for attack in os.listdir(attacks_path):
        # Skip over any file that is not a numpy file or the labels file
        if "labels" in attack or ".npy" not in attack:
            continue
        images_by_severity = []
        # Images are stored in 0-255 pixel range in the format (N, h, w, c)
        # where c is number of channels.
        # c needs to become the second value in the images' shape.
        attack_images = np.transpose(np.load(f"{attacks_path}/{attack}"),
                                     (0, 3, 1, 2)) / 255.
        for i in range(0, len(all_labels), total):
            images_by_severity.append(attack_images[i:i + total])

        # Batch final data by severity
        test_data_by_severity = [[], [], [], [], []]
        for severity in range(5):
            images, labels = utils.batch_data(batch_size,
                                              images_by_severity[severity],
                                              labels_by_severity[severity])
            for batch_index in range(len(labels)):
                test_data_by_severity[severity].append((
                    torch.FloatTensor(images[batch_index]),
                    torch.LongTensor(labels[batch_index])))

        # Actually run the attacks consecutively for each model
        # Avoids reloading data
        for model in mdls:
            desc = f"{model.name} {data} {attack[0:-4]}"
            accs = []
            # Mean of 3 clean accuracy runs is recorded
            # as a sort of severity 0 attack
            for _ in range(3):
                accs.append(acc(model))
            results = [st.mean(accs)]
            for severity in range(5):
                attacked = np.around(acc(model,
                                         test_data=test_data_by_severity[severity],
                                         desc=f"{desc} severity {severity + 1}"), 1)
                results.append(attacked)

            # Log the results in the corresponding file along with average and
            # standard deviation across all severities
            f = open(f"Results/{desc}.txt", 'w')
            for sev, accur in zip([0, 1, 2, 3, 4, 5], results):
                f.write(f"Severity {sev:<5}: {accur:4.1f}%\n")
            f.write(f"Mean:   {np.around(st.mean(results), 1)}\n")
            f.write(f"Stddev: {np.around(st.stdev(results), 1)}\n")
            f.close()


def linfpgd(epsilons=None):
    if epsilons is None:
        epsilons = np.around(np.linspace(0.002, 0.004, num=2), 3)
    return Attack("LinfPGD", LinfPGD(), epsilons=epsilons)


def l1pgd(epsilons=None):
    if epsilons is None:
        epsilons = np.around(np.linspace(12, 24, num=2), 3)
    return Attack("L1PGD", L1PGD(), epsilons=epsilons)


def l2pgd(epsilons=None):
    if epsilons is None:
        epsilons = np.around(np.linspace(0.1, 0.2, num=2), 3)
    return Attack("L2PGD", L2PGD(), epsilons=epsilons)


def gaussian_blur(lp_distance, epsilons=None):
    if epsilons is None:
        epsilons = np.around(np.linspace(5, 25, num=4), 3)
    return Attack("Gaussian Blur",
                  GaussianBlurAttack(distance=LpDistance(lp_distance)),
                  epsilons)


def clipping_aware_gaussian_noise(epsilons=None):
    if epsilons is None:
        epsilons = np.around(np.linspace(5, 20, num=4), 3)
    return Attack("L2 Clipping Aware Additive Gaussian Noise",
                  L2ClippingAwareAdditiveGaussianNoiseAttack(),
                  epsilons)


def clipping_aware_uniform_noise(epsilons=None):
    if epsilons is None:
        epsilons = np.around(np.linspace(5, 20, num=4), 3)
    return Attack("L2 Clipping Aware Additive Uniform Noise",
                  L2ClippingAwareAdditiveUniformNoiseAttack(),
                  epsilons)


def uniform_noise(epsilons=None):
    if epsilons is None:
        epsilons = np.around(np.linspace(5, 20, num=4), 3)
    return Attack("L2 Additive Uniform Noise",
                  L2AdditiveUniformNoiseAttack(),
                  epsilons)


def contrast_reduction(epsilons=None):
    if epsilons is None:
        epsilons = np.around(np.linspace(5, 25, num=5), 3)
    return Attack("L2 Contrast Reduction",
                  L2ContrastReductionAttack(),
                  epsilons)
