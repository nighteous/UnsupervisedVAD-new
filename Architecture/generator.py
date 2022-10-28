from collections import OrderedDict
import torch
import torch.nn as nn

class AE(nn.Module):

    def __init__(self, list):
        super().__init__()

        # Traversing through the list to form both encoder and decoder layers

        if list[0] > 2048 or list[-1] > 2048:
            # for resnext features
            raise Exception("Feauture input or output must be of 2048")

        encoder_layers = []
        for i in range(len(list)):
            if list[i] < list[i+1]:
                break
            
            layerName = [('linear{}'.format(i + 1), nn.Linear(list[i], list[i+1]),), ('batchnorm{}'.format(i +1), nn.BatchNorm1d(list[i+1])), ('relu{}'.format(i +1), nn.ReLU())]

            encoder_layers.extend(layerName)


        decoder_layers = []
        for i in range( (len(list)) // 2, len(list)):

            try:
                layerName = [('linear{}'.format(i + 1), nn.Linear(list[i], list[i+1]),), 
                            ('batchnorm{}'.format(i + 1), nn.BatchNorm1d(list[i+1])), 
                            ('relu{}'.format(i + 1), nn.ReLU())
                    ]
                decoder_layers.extend(layerName)

            except:
                pass

        self.encoder = nn.Sequential(OrderedDict(encoder_layers))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))

    def forward(self, img):
        encoded = self.encoder(img)
        decoded = self.decoder(encoded)

        return decoded
