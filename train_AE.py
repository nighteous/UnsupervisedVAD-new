import argparse
from collections import OrderedDict
from pickle import TRUE
import time

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
import wandb

def options():
    # Arguments to pass from terminal to make our lives easier
    parser = argparse.ArgumentParser(description='Training of AE')
    parser.add_argument('--arch',
                        dest='arch',
                        type=str,
                        help="Architecture of the model Eg: 1,2,3,4")

    parser.add_argument('--lr', dest='lr', type=float, default=3e-4,help="Learning rate of model")
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help="Number of Epochs to train model")
    parser.add_argument('--n', dest='n_trails', type=int, default=5, help ="Number of times to train the model")
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=32, help="Batch size of data")
    parser.add_argument('--weight-decay', dest='weightdecay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.6, help ="Dropout layer value, default: 0.6")
    parser.add_argument('--optimizer', dest='optimizer_name', type=str, default='SGD', help="Optimizer to be used\n1. SGD\n2. Adam")
    parser.add_argument('--count', dest='count', type=int, default=1, help="Count of which model (only needed for wandb)")

    return parser.parse_args()


args = options()

# wandb initialization
wandb.init(project="autoencoder{}".format(args.count), config = args)


FeatsPath = "/shared/home/v_varenyam_bhardwaj/local_scratch/Dataset/UCFResnext/different_dataset_splits/"
train_normal_feats = np.load(FeatsPath + "normal_train_set_video_features.npy")


device = torch.device("cuda:0")

# Architecture of AE
model_arch = [int(i) for i in args.arch.split(',')]
model = AE(model_arch, 0.6).to(device)

wandb.watch(model, log_freq = 100)


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



if args.optimizer_name.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
elif args.optimizer_name.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

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

startTime = time.time()

for epoch in range(epochs):

    am_mil = __import__('utils').AverageMeter()
    train_loss = 0

    print("Epoch {}".format(epoch+1))
    result.write("Epoch {}\n".format(epoch+1))

    # on train set
    model.train()
    for idx, img in tqdm(enumerate(train_dataset)):

        img = img.to(device)
        

        prediction = model(img)

        # Previous method of calculating loss
        # loss = criterion(prediction, img)
        # train_loss += loss.item()*img.size(0)

        loss = torch.abs(prediction - img)
        loss = loss.sum(1).mean(0)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        am_mil.update(loss.item())


    loss_values_train.append(am_mil.avg)

    wandb.log({"Train Loss": am_mil.avg})

    print("Train Loss: {}".format(am_mil.avg))
    result.write("Train Loss: {}\n".format(am_mil.avg))

endTime = time.time()

print("Time taken by {} epochs: {}".format(epochs, (endTime - startTime)))
result.write("Time taken by {} epochs: {}".format(epochs, (endTime - startTime)))

np.save('Losses/AE/Loss_Train_{}.npy'.format(filesuffix), loss_values_train)
torch.save(model.state_dict(), 'SavedModels/AE/{}.pth'.format(filesuffix))
result.close()