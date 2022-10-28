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

    parser.add_argument('--optimizer', dest='optimizer_name', type=str, default='Adam', help="Optimizer to be used -> Doesn't work")


    return parser.parse_args()


args = options()

FeatsPath = "/shared/home/v_varenyam_bhardwaj/local_scratch/Dataset/FeaturesResnext/"
train_normal_feats = np.load(FeatsPath + "normal_train_set_video_features.npy")


device = torch.device("cuda:0")

# Architecture of AE
model_arch = [int(i) for i in args.arch.split(',')]
model = AE(model_arch).to(device)

model_name = "AE_"
for i in model_arch:
    model_name += str(i) + "_"

model_name = model_name[:-1]

torch.manual_seed(18)
transform = transforms.Compose([
    transforms.ToTensor()
    ])


train_dataset = DataLoader(train_normal_feats,
                           batch_size = args.batchsize,
                           shuffle = True,
                           num_workers=8
               )

# IDK how to decide this one
optimizer_name="Adam"
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weightdecay)
criterion = nn.MSELoss()

filesuffix = "{}_opt_{}_ep_{}_lr_{}_wgd_{}_bs_{}".format(model_name, args.optimizer_name, args.epochs, args.lr ,args.weightdecay, args.batchsize) 

try:
    result = open("Outputs/AE/Output_Train_{}.txt".format(filesuffix), "w")
except Exception as e:
    print(e)
    exit()

result.write("Learning Rate: {}\n".format(args.lr))
result.write("Weight Decay: {}\n".format(args.weightdecay))
result.write("Epochs: {}\n".format(args.epochs))
result.write("Batch Size: {}\n".format(args.batchsize))
result.write(str(model) + "\n")
result.write(str(optimizer) + "\n")

best_loss = 1e10
epochs = args.epochs

loss_values_train = []
loss_values_test = []


for epoch in range(epochs):

    train_loss = 0

    print("Epoch {}".format(epoch+1))
    result.write("Epoch {}\n".format(epoch+1))

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
    result.write("Train Loss: {}\n".format(train_loss))


np.save('Losses/AE/Loss_Train_{}.npy'.format(filesuffix), loss_values_train)
torch.save(model.state_dict(), 'SavedModels/AE/{}.pth'.format(filesuffix))
result.close()