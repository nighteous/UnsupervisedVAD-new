import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import *
import os
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import config
import torch.nn.functional as F
import math
from utils import AverageMeter


class Learner(nn.Module):
    def __init__(self, input_dim=1024, drop_p=0.0):
        super(Learner, self).__init__()

        self.f1 = nn.Linear(1024, 512)
        self.f2 = nn.Linear(512, 256)
        self.f3 = nn.Linear(256, 256)
        self.f4 = nn.Linear(256, 512)
        self.f5 = nn.Linear(512, 1024)

        self.do1 = nn.Dropout(0.6)
        self.do2 = nn.Dropout(0.6)
        self.do3 = nn.Dropout(0.6)
        self.do4 = nn.Dropout(0.6)
        self.do5 = nn.Dropout(0.6)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.do1(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.do2(x)
        x = self.f3(x)
        x = F.relu(x)
        x = self.do3(x)
        x = self.f4(x)
        x = F.relu(x)
        x = self.f5(x)
        return x


def train(epoch, model, optimizer, normal_train_loader, anomaly_train_loader, device, max_epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loader = tqdm(zip(normal_train_loader, anomaly_train_loader),
                  total=min(len(normal_train_loader), len(anomaly_train_loader)),
                  bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}")

    # print(len(anomaly_train_loader.dataset))
    # exit()
    am_mil = AverageMeter()
    for batch_idx, (normal_data, anomaly_data) in enumerate(loader):
        normal_inputs = normal_data[0]
        normal_index = normal_data[1]

        anomaly_inputs = anomaly_data[0]
        anomaly_index = anomaly_data[1]

        bs = len(normal_inputs)

        # inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)

        normal_inputs = normal_inputs.to(device)

        normal_inputs = normal_inputs.view(-1, 1024)

        outputs = model(normal_inputs)  # (bs x 2 x 32) x 1

        loss = torch.abs(outputs - normal_inputs)

        loss = loss.sum(1).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train_loss_n += loss_cont_n.item()
        # train_loss_a += loss_cont_a.item()

        # print(loss.item(), '<<<<<<<<<<<<')
        am_mil.update(loss.item())

        loader.set_description(
            "E{}/{}, loss:{:2.4f}".format(
                epoch, max_epoch, am_mil.avg))


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


def main(max_epoch=75):
    normal_train_dataset = Normal_Loader_indexed(is_train=1)
    normal_test_dataset = Normal_Loader(is_train=0)

    anomaly_train_dataset = Anomaly_Loader_indexed(is_train=1)
    anomaly_test_dataset = Anomaly_Loader(is_train=0)

    bs = 32

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=bs, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=bs, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

    device = 'cuda:0'  # if torch.cuda.is_available() else 'cpu'

    model = Learner(input_dim=config.INPUT_DIM).to(device)  # 0.6
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-1)

    best_auc = 0

    for epoch in range(0, max_epoch):
        train(epoch, model, optimizer, normal_train_loader, anomaly_train_loader, device, max_epoch)

        test_auc = test_abnormal(epoch, model, normal_test_loader, anomaly_test_loader, device)
        if test_auc > best_auc:
            best_auc = test_auc
        print('\ntest AUC: %.2f, best AUC: %.2f' % (test_auc, best_auc))
    return best_auc


if __name__ == "__main__":
    main()
