import torchvision.models as models
import adversarial_attacks
from V1 import lnp, pcbc

import time
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as f
from torch.optim.lr_scheduler import StepLR


class Model:
    def __init__(self, agent_name, agent, data):
        self.directory = "Models"
        self.name = f"{agent_name} {data.name}"
        self.data = data
        self.agent = agent

        # Checks whether model must repeat the channels value three times
        # Only an issue when the input data is
        # grayscale and the CNN is mobilenetv2 as
        # it cannot accept 1-channel data
        self.repeat = False
        if agent_name == "clean" and data.channels == 1:
            self.repeat = True

    def save(self):
        torch.save(self.agent.state_dict(), f"{self.directory}/{self.name}.pyt")

    def load_for_eval(self):
        self.agent.load_state_dict(torch.load(f"{self.directory}/{self.name}.pyt"))
        return self.agent.eval()

    def print_name(self):
        time.sleep(1)
        print(self.name)
        time.sleep(1)

    # Trains the model with the data it is supplied at initialisation
    def train(self, print_loss=True):
        self.print_name()
        optimizer = optim.SGD(self.agent.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=self.data.step_size, gamma=0.1)
        for epoch in range(self.data.epochs):
            total_loss = 0
            total = 0
            for data in tqdm(self.data.train, desc=f"Epoch {epoch + 1}: "):
                images, labels = data
                # Turn grayscale data into 3-channel data
                if self.repeat:
                    images = images.repeat(1, 3, 1, 1)
                images = images.to("cuda:0")
                labels = labels.to("cuda:0")

                # Reset gradient
                self.agent.zero_grad()
                output = self.agent(images)

                # Calculate loss and backpropagate to update model
                loss = f.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss
                total += 1
            batch_loss = total_loss / total
            if print_loss:
                print(batch_loss)
                time.sleep(1)
            scheduler.step()
        self.save()

    # Attacks the model with a list of foolbox attacks and logs the results
    def adv_attack(self, attacks):
        self.print_name()
        model = adversarial_attacks.pytorch_model(self.load_for_eval())

        for attack in attacks:
            accuracy = None
            batches = 0
            for data in tqdm(self.data.test, desc=f"{attack.name}: "):
                images, labels = data
                if self.repeat:
                    images = images.repeat(1, 3, 1, 1)
                images = images.to("cuda:0")
                labels = labels.to("cuda:0")
                raw, clipped, success = attack(model, images, labels)
                robust_accuracy = 1 - success.float().mean(axis=-1)
                # If it is the first time calculating accuracy,
                # assign it to the total accuracy. Otherwise, add it to it
                if accuracy is None:
                    accuracy = robust_accuracy
                else:
                    accuracy += robust_accuracy
                batches += 1

                # Log results in appropriate file
                file = open(f"Results/{self.name} {attack.name}", "w")
                for eps, acc in zip(attack.epsilons, accuracy):
                    file.write(f"Epsilon {eps:<10}: "
                               f"{acc.item() * 100 / batches:4.1f}%\n")
                file.close()


def mobnetv2(data):
    return Model("mobnetv2",
                 models.mobilenet_v2(
                     **{"num_classes": data.classes}).to("cuda:0"), data)


def lnpnet(data, sig=(1.5, 2), lam=(4, 7)):
    return Model(f"lnpnet sig={sig} lam={lam}",
                 lnp.VOneMobileNet(data.classes,
                                   data.channels,
                                   sig=sig,
                                   lam=lam).to("cuda:0"), data)


def pcbcnet(data, iterations=3, sig=1, lam=1.5, sigp=1):
    return Model(f"pcbcnet t{iterations} sig{sig} lam{lam} sigp{sigp}",
                 pcbc.VOneMobileNet(data.classes,
                                    data.channels,
                                    iterations,
                                    sig,
                                    lam,
                                    sigp).to("cuda:0"), data)
