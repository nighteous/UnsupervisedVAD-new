import torch
import torch.nn as nn

class AE(nn.Module):
    """
    AutoEncoder class implemented according to the paper
        FC[2048, 1024, 512, 256, 512, 1024, 2048]
    """

    def __init__(self):
        super().__init__()

        img_size = 2048

        self.encoder = nn.Sequential(

                # nn.Linear(2048, 1024),
                # nn.ReLU(),

                nn.Linear(1024, 512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.ReLU(),

                nn.Linear(256, 128),
                nn.ReLU()
        )

        self.decoder = nn.Sequential(
                nn.Linear(128, 256),
                nn.LeakyReLU(),

                nn.Linear(256, 512),
                nn.LeakyReLU(0.3),

                nn.Linear(512, 1024),
                # nn.LeakyReLU(0.3),

                # nn.Linear(1024, 2048),

                nn.Sigmoid()
        )

    def forward(self, img):
        encoded = self.encoder(img)
        decoded = self.decoder(encoded)

        return decoded
