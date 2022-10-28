import argparse
from collections import OrderedDict
from pickle import TRUE

from Architecture.generator import AE

import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List


def options():
    # Arguments to pass from terminal to make our lives easier
    parser = argparse.ArgumentParser(description='Training of AE')
    parser.add_argument('--arch', 
                        dest='arch', 
                        type=str, 
                        help="Architecture of the model Eg: 1,2,3,4")

    parser.add_argument('--lr', dest='lr', type=float, default=3e-4,help="Learning rate of model")

    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help="Number of Epochs to train model")

    parser.add_argument('--batch-size', dest='batchsize', type=int, default=128, help="Batch size of data")

    parser.add_argument('--weight-decay', dest='weightdecay', type=float, default=1e-4, help="Weight decay")


    return parser.parse_args()


args = options()

train_normal_feats = np.load('~/local_scratch/Dataset/FeaturesResnext/normal_train_set_video_features.npy')
device = torch.device("cuda:0")

# Architecture of AE
arch = [int(i) for i in args.arch.split(',')]
model = AE(arch).to(device)


torch.manual_seed(18)
transform = transforms.Compose([
    transforms.ToTensor()
    ])


train_dataset = DataLoader(train_normal_feats,
                           batch_size = args.batchsize,
                           shuffle = True,
                           num_workers=8
               )

# test_dataset_normal = DataLoader(test_feats_normal, batch_size = 128, shuffle = False)
# print(test_dataset_normal.dataset.shape)
# test_dataset_anomalous = DataLoader(test_feats_anomolous, batch_size = 128, shuffle = False)
# model 1

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weightdecay)
criterion = nn.MSELoss()

best_loss = 1e10
epochs = args.epochs

loss_values_train = []
loss_values_test = []

for epoch in range(epochs):

    train_loss = 0

    print("Epoch {}".format(epoch+1))

    # on train set
    model.train()
    for idx, img in tqdm(enumerate(train_dataset)):

        img = img.to(device)
        optimizer.zero_grad()

        prediction = model(img)

        loss = criterion(prediction, img)
        train_loss += loss.item()*img.size(0)

        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataset)
    loss_values_train.append(train_loss)
    print("Train Loss: {}".format(train_loss))

model_name = "AE_"
for i in arch:
    model_name += str(i) + "_"

model_name = model_name[:-1]


np.save('Losses/AE/{}.npy'.format(model_name), loss_values_train)
torch.save(model.state_dict(), 'SavedModels/AE/{}.pth')
