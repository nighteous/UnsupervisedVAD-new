from Architecture.generator import *
from Architecture.discriminator import *
import os
import numpy as np
import torch
from torch.optim import RMSprop
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
from tqdm import tqdm

path = '/shared/home/v_varenyam_bhardwaj/local_scratch/Dataset/UCF-Crime/all_rgbs'
num_output_features = len(os.listdir(path))

x = os.listdir(path)
feats = []

for i in x:
    z = os.listdir(os.path.join(path, i))
    for j in z:
        feat_npy = np.load(os.path.join(os.path.join(path, i), j))
        feat_npy = np.array(feat_npy, dtype = np.float32)

        for k in feat_npy:
            feats.append(k)

del feat_npy # to delete feat_npy because we dont need it anymore

print("Shape of features is {}".format(np.array(feats).shape))