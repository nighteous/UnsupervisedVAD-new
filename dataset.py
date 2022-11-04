import torch
import config
import random
import os
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, features) -> None:
        super().__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index]



path_to_splits = 'splits/'
# For loading features from i3d
class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, path=config.DATASET_DIR):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = path_to_splits + 'train_normal.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = path_to_splits + 'test_normalv2.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            # self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(
                self.path + 'all_rgbs', self.data_list[idx][:-1] + '.npy'))
            # flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
            # concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return rgb_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(
                self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            rgb_npy = np.load(
                os.path.join(
                    self.path + 'all_rgbs',
                    name + '.npy'))
            # flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            # concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return rgb_npy, gts, frames


class Normal_Loader_indexed(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, path=config.DATASET_DIR):
        super(Normal_Loader_indexed, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = path_to_splits + 'train_normal.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = path_to_splits + 'test_normalv2.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            # self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(
                self.path + 'all_rgbs', self.data_list[idx][:-1] + '.npy'))
            # flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
            # concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return rgb_npy, idx
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(
                self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            rgb_npy = np.load(
                os.path.join(
                    self.path + 'all_rgbs',
                    name + '.npy'))
            # flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            # concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return rgb_npy, gts, frames


class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, path=config.DATASET_DIR):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = path_to_splits + 'train_anomaly.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = path_to_splits + 'test_anomalyv2.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(
                self.path + 'all_rgbs', self.data_list[idx][:-1] + '.npy'))
            # flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
            # concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return rgb_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(
                self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(
                os.path.join(
                    self.path + 'all_rgbs',
                    name + '.npy'))
            # flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            # concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return rgb_npy, gts, frames


class Anomaly_Loader_indexed(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """

    def __init__(self, is_train=1, path=config.DATASET_DIR):
        super(Anomaly_Loader_indexed, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = path_to_splits + 'train_anomaly.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = path_to_splits + 'test_anomalyv2.txt'
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(
                self.path + 'all_rgbs', self.data_list[idx][:-1] + '.npy'))
            # flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
            # concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return rgb_npy, idx
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(
                self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(
                os.path.join(
                    self.path + 'all_rgbs',
                    name + '.npy'))
            # flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            # concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return rgb_npy, gts, frames


if __name__ == '__main__':
    loader2 = Normal_Loader(is_train=0)
    print(len(loader2))
    # print(loader[1], loader2[1])
