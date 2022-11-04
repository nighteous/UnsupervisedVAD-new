import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import datetime
import cv2
import matplotlib.pyplot as plt
import copy



def norm_minmax(x):
    """
    min-max normalization of numpy array
    """
    return (x - x.min()) / (x.max() - x.min())


def plot_tensor(t):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    for i in range(len(t)):
        ti_np = t[i].cpu().detach().numpy().squeeze()
        ti_np = norm_minmax(ti_np)
        if len(ti_np.shape) > 2:
            ti_np = ti_np.transpose(1, 2, 0)
        plt.subplot(1, len(t), i + 1)
        plt.imshow(ti_np)
    plt.show()


def tensor2img(t, normlize=True):
    img = t.cpu().detach().numpy()
    if normlize:
        img = norm_minmax(img)
    img = img.transpose(1, 2, 0)
    return img




def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class Logger:
    def __init__(self, var_names=None, format=None, args=None, print_args=False, save_dir=None):


        if args is None:
            self.dir = save_dir
        else:
            self.dir = args.save_dir
        self.var_names = var_names
        self.format = format
        self.vars = []

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        file = open(self.dir + '/log.txt', 'w')
        file.write('Log file created on ' + str(datetime.datetime.now()) + '\n\n')


        if args is not None:
            dict = {}
            for arg in vars(args):
                dict[arg] = str(getattr(args, arg))
                print(arg, str(getattr(args, arg)))

            for d in sorted(dict.keys()):
                file.write(d + ' : ' + dict[d] + '\n')
            file.write('\n')
            file.close()


    def store(self, vars, log=False):
        self.vars = self.vars + vars
        if log:
            self.log()

    def log(self):

        vars = self.vars
        file = open(self.dir + '/log.txt', 'a')
        st = ''
        for i in range(len(vars)):
            st += self.var_names[i] + ': ' + self.format[i] % (vars[i]) + ', '
        st += 'time: ' + str(datetime.datetime.now()) + '\n'
        file.write(st)
        file.close()
        self.vars = []

    def store_text(self, text):
        file = open(self.dir + '/log.txt', 'a')
        st = ''
        file.write(text + '\n')
        file.close()


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(args, acc_current, acc_best, net, optimizer, epoch, net_best):
    if acc_current > acc_best:
        print('Saving best model...')
        state = {
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': acc_current,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(args.save_dir, 'model_best.pt'))
        acc_best = acc_current
        net_best = copy.deepcopy(net)
    if args.trace:
        print('Saving for trace...')
        state = {
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': acc_current,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(args.save_dir, 'model_' + str(epoch) + '.pt'))
    if epoch + 2 in args.lr_decay_epochs:
        print('Saving for trace2...')
        state = {
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': acc_current,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(args.save_dir, 'model_' + str(epoch) + '.pt'))
    return acc_best, net_best
