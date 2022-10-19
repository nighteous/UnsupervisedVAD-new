import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator model made as per paper [https://arxiv.org/pdf/2203.03962.pdf]
    Fully Connected Layer
    FC[2048, 512, 32, 1]
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU(0.3)
        self.sigmoid=nn.Sigmoid()

    def forward(self, inp):

        out = self.fc1(inp)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out