import argparse

from Architecture.generator import AE

from dataset import *

import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryROC
from torchmetrics import ROC
from torchmetrics.functional.classification import binary_roc
from sklearn import metrics

def options():
    # Arguments to pass from terminal to make our lives easier
    parser = argparse.ArgumentParser(description='Testing of AE')
    parser.add_argument('--arch',
                        dest='arch',
                        type=str,
                        default='idk',
                        help="Architecture of the model Eg: 1,2,3,4")

    parser.add_argument('--lr', dest='lr', type=float, default=3e-4,help="Learning rate of model")

    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help="Number of Epochs to train model")

    parser.add_argument('--batch-size', dest='batchsize', type=int, default=128, help="Batch size of data")

    parser.add_argument('--weight-decay', dest='weightdecay', type=float, default=1e-4, help="Weight decay")

    parser.add_argument('--optimizer', dest='optimizer_name', type=str, default='SGD', help="Optimizer to be used -> Doesn't work")

    return parser.parse_args()


args = options()


# Loading Features
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

# Defining Device
try:
    device = torch.device("cuda:0")
except:
    device = torch.device("cpu")

# from AutoEncoder import AE
arch = [int(i) for i in args.arch.split(',')]
model = AE(arch).to(device)

model_name = "AE_"
for i in arch:
    model_name += str(i) + "_"
model_name = model_name[:-1]


filesuffix = "{}_opt_{}_ep_{}_lr_{}_wgd_{}_bs_{}".format(model_name, args.optimizer_name, args.epochs, args.lr ,args.weightdecay, args.batchsize)


try:
    result = open("Outputs/AE/Output_Test_{}.txt".format(filesuffix), "w")
except Exception as e:
    print(e)
    exit()



model.load_state_dict(torch.load('SavedModels/AEi3d/{}.pth'.format(filesuffix)))

device = "cuda:3" if torch.cuda.is_available() else "cpu"
# print("Device in use is {}".format(device))
torch.manual_seed(18)
model = model.to(device)

# Not used (needed for input to file)
optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay=args.weightdecay)

result.write(str(model) + "\n")
result.write(str(optimizer) + "\n")


transform = transforms.Compose([
    transforms.ToTensor()
    ])

criterion = nn.MSELoss()

best_loss = 1e10
epochs = 1

loss_values_test = []
normal_predictions = []
anomalous_predictions = []


def test_abnormal(epoch, model, normal_test_loader, anomaly_test_loader, device):
    model.eval()
    auc = 0
    all_gt_list = []
    all_score_list = []

    error_a = 0
    error_n = 0
    counter = 0

    with torch.no_grad():
        for i, (data) in enumerate(anomaly_test_loader):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device(device))
            score = model(inputs)
            score = torch.abs(score - inputs).sum(1)

            error_a += score.mean()
            counter += 1

            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.linspace(0, frames[0], 33)

            for j in range(32):
                score_list[int(step[j]):(int(step[j + 1]))] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = gts[k * 2]
                e = min(gts[k * 2 + 1], frames)
                gt_list[s - 1:e] = 1

            all_gt_list.extend(gt_list)
            all_score_list.extend(score_list)

        print('error anormal: %.3f' % (error_a / counter))
        counter = 0

        for i, (data2) in enumerate(normal_test_loader):
            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device(device))
            # score2 = model(inputs2)

            score2 = model(inputs2)
            score2 = torch.abs(score2 - inputs).sum(1)

            error_n += score2.mean()
            counter += 1

            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.linspace(0, frames2[0], 33)
            for kk in range(32):
                score_list2[int(step2[kk]):(int(step2[kk + 1]))] = score2[kk]
            gt_list2 = np.zeros(frames2[0])

            all_gt_list.extend(gt_list2)
            all_score_list.extend(score_list2)

        auc = metrics.roc_auc_score(all_gt_list, all_score_list)
        print('error normal: %.3f' % (error_n / counter))
        # print(f"The AUC SCORE is {auc}")

    return auc * 100

for epoch in range(epochs)
