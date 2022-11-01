from torch.utils.data import DataLoader
from Architecture.generator import AE
from utils import AverageMeter

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import *

def options():
    # Arguments to pass from terminal to make our lives easier
    parser = argparse.ArgumentParser(description='Training of AE')
    parser.add_argument('--arch',
                        dest='arch',
                        type=str,
                        help="Architecture of the model Eg: 1,2,3,4")

    parser.add_argument('--lr', dest='lr', type=float, default=3e-4,help="Learning rate of model")
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help="Number of Epochs to train model")
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=32, help="Batch size of data")
    parser.add_argument('--weight-decay', dest='weightdecay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--optimizer', dest='optimizer_name', type=str, default='SGD', help="Optimizer to be used -> Doesn't work")

    return parser.parse_args()

args = options()

device = torch.device("cuda:0")

# Architecture of AE
model_arch = [int(i) for i in args.arch.split(',')]
model = AE(model_arch).to(device)

model_name = "AE_"
for i in model_arch:
    model_name += str(i) + "_"

model_name = model_name[:-1]

torch.manual_seed(18)

# optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weightdecay)
criterion = nn.MSELoss()

filesuffix = "{}_opt_{}_ep_{}_lr_{}_wgd_{}_bs_{}".format(model_name, args.optimizer_name, args.epochs, args.lr ,args.weightdecay, args.batchsize)

try:
    result = open("Outputs/AE/Output_Train_i3d_{}.txt".format(filesuffix), "w")
except Exception as e:
    print(e)
    exit()

result.write("Learning Rate: {}\n".format(args.lr))
result.write("Weight Decay: {}\n".format(args.weightdecay))
result.write("Epochs: {}\n".format(args.epochs))
result.write("Batch Size: {}\n".format(args.batchsize))
result.write(str(model) + "\n")

best_loss = 1e10
epochs = args.epochs

loss_values_train = []
loss_values_test = []

normal_train_dataset = Normal_Loader_indexed(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)

anomaly_train_dataset = Anomaly_Loader_indexed(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)

normal_train_loader = DataLoader(normal_train_dataset,
                                 batch_size=args.batchsize,
                                 shuffle=True
                            )

normal_test_loader = DataLoader(normal_test_dataset,
                                batch_size=1,
                                shuffle=True
                            )

anomaly_train_loader = DataLoader(anomaly_train_dataset,
                                 batch_size=args.batchsize,
                                 shuffle=True
                            )

anomaly_test_loader = DataLoader(anomaly_test_dataset,
                                batch_size=1,
                                shuffle=True
                            )

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
result.write(str(optimizer) + "\n")

epochs = args.epochs

model.train()
for epoch in range(epochs):
    print("\nEpoch: {}".format(epoch))
    result.write("Epoch {}\n".format(epoch+1))

    train_loss = 0
    correct = 0
    total = 0

    am_util = AverageMeter()

    for idx, data in tqdm(enumerate(normal_train_loader)):
        data = data[0].to(device)
        data = data.view(-1, 1024)

        output = model(data)

        loss = torch.abs(output - data)
        loss = loss.sum(1).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print("Train Loss: {}".format(train_loss))
    result.write("Train Loss: {}\n".format(train_loss))

torch.save(model.state_dict(), "./SavedModels/AEi3d//{}.pth".format(filesuffix))
result.close()

