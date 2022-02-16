import datasets
import models
from adversarial_attacks import acc
from adversarial_attacks import clipping_aware_gaussian_noise, \
    clipping_aware_uniform_noise, \
    gaussian_blur, \
    l1pgd, \
    l2pgd, \
    linfpgd, \
    contrast_reduction, \
    uniform_noise, \
    dashc


def attack(mdls, attacks, train=False):
    for model in mdls:
        if train:
            model.train()
        acc(model)
        model.adv_attack(attacks)


if __name__ == "__main__":
    cifar10 = datasets.CIFAR10(batch_size=100, epochs=25)
    cifar100 = datasets.CIFAR100(batch_size=100, step_size=20)

    mdls = [
        models.mobnetv2(data=cifar10),
        models.mobnetv2(data=cifar100),
        models.lnpnet(data=cifar10),
        models.lnpnet(data=cifar100, sig=(1, 3), lam=(5, 10)),
        models.pcbcnet(data=cifar10, sig=2, lam=1.5, sigp=2),
        models.pcbcnet(data=cifar100),
    ]

    attacks = [
        clipping_aware_uniform_noise(),
        l1pgd(),
        l2pgd()
    ]

    attack(mdls, attacks, train=True)
