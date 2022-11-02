from collections import OrderedDict
import torch
import torch.nn as nn

class AE(nn.Module):

    def __init__(self, list, p):
        super().__init__()

        # Traversing through the list to form both encoder and decoder layers

        encoder_layers = []
        for i in range(len(list)):
            if list[i] < list[i+1]:
                break
            
            layerName = [('linear{}'.format(i + 1), nn.Linear(list[i], list[i+1]),), 
                        ('batchnorm{}'.format(i + 1), nn.BatchNorm1d(list[i+1])), 
                        ('relu{}'.format(i + 1), nn.ReLU()),
                        ('dropout{}'.format(i + 1), nn.Dropout(p))]

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
