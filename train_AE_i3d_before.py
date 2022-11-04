from torch.utils.data import DataLoader
from Architecture.generator_before import AE
from utils import AverageMeter

from sklearn import metrics
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
    parser.add_argument('--n', dest='n_trails', type=int, default=5, help ="Number of times to train the model")
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=32, help="Batch size of data")
    parser.add_argument('--weight-decay', dest='weightdecay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--dropout', dest='dropout', type=float, default=0, help ="Dropout layer value")
    parser.add_argument('--optimizer', dest='optimizer_name', type=str, default='SGD', help="Optimizer to be used\n1. SGD\n2. Adam")

    return parser.parse_args()

args = options()

device = torch.device("cuda:6")
torch.manual_seed(18)

# Architecture of AE
model_arch = [int(i) for i in args.arch.split(',')]
model = AE(model_arch, args.dropout).to(device)

model_name = "AE_"
for i in model_arch:
    model_name += str(i) + "_"

model_name = model_name[:-1]

# optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weightdecay)
criterion = nn.MSELoss()

filesuffix = "{}_opt_{}_ep_{}_lr_{}_wgd_{}_bs_{}_dropout_{}_before".format(model_name, args.optimizer_name, args.epochs, args.lr ,args.weightdecay, args.batchsize, args.dropout)

try:
    result = open("Outputs/AE/i3d/BNBefore/Output_Train_i3d_{}.txt".format(filesuffix), "w")
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

if args.optimizer_name.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
elif args.optimizer_name.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

result.write(str(optimizer) + "\n")


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
        result.write('error normal: %.3f' % (error_n / counter))
        # print(f"The AUC SCORE is {auc}")

    return auc * 100
all_auc = []

for j in range(args.n_trails):
    print("\nN Trail: {}".format(j))
    result.write("\nN Trail: {}".format(j))


    epochs = args.epochs
    best_auc = 0
    
    model = AE(model_arch, args.dropout).to(device)
    for epoch in range(epochs):
        print("\nEpoch: {}".format(epoch))
        result.write("\nEpoch {}".format(epoch+1))

        model.train()
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

        test_auc = test_abnormal(epoch, model, normal_test_loader, anomaly_test_loader, device)


        if test_auc > best_auc:
            best_auc = test_auc
        print('\ntest AUC: %.2f, best AUC: %.2f' % (test_auc, best_auc))
        result.write('\ntest AUC: %.2f, best AUC: %.2f' % (test_auc, best_auc))


    print("\nBest AUC: {}\n".format(best_auc))
    result.write("\nBest AUC: {}\n".format(best_auc))

    all_auc.append(best_auc)

aucs = np.array(all_auc)
print('AUC stats over %d runs : mean: %.2f, std: %.2f'% (args.n_trails, aucs.mean(), aucs.std()))

torch.save(model.state_dict(), "./SavedModels/AEi3d/{}.pth".format(filesuffix))
result.close()

