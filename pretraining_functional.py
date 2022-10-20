import torch
from torch.nn import MSELoss
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from Architecture.generator import AE
from Architecture.discriminator import Discriminator

from dataset import CustomDataset

import os
from typing import List
import numpy as np
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_features(path: str) -> np.ndarray:
    """
    Loads features to a numpy array and returns it
    Args
        Input: path to features directory
        Output: Numpy array of features
    """

    listOfDirectories = os.listdir(path)
    features= []

    for i in listOfDirectories:
        listOfSubDirectories = os.listdir(os.path.join(path, i))

        for j in listOfSubDirectories:
            feat_npy = np.load(os.path.join(os.path.join(path, i), j))
            feat_npy = np.array(feat_npy, dtype = np.float32)

            for k in feat_npy:
                features.append(k)

    features = np.array(features)

    return features

def data_loader(features: np.ndarray, batch_size: int, shuffle: bool):
    """
    Takes in array of features and returns DataLoader of given batch size
    """
    dataset = CustomDataset(features)
    trainDataLoader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    return trainDataLoader

def train_generator(trainLoader: DataLoader, model:AE, optimizer: Optimizer ,lossFunctionG, epochs: int):
    """
    Training loop for Generator
    """

    outputs = []
    losses = []
    epoch_losses = []

    for epoch in range(epochs):

        running_loss = 0.
        print("Epoch No.: {}".format(epoch + 1))

        for i, data in tqdm(enumerate(trainLoader)):
            data = data.to(device)
            reconstructed = model.forward(data)

            loss = lossFunctionG(reconstructed, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(trainLoader)

        print("Loss: {}".format(epoch_loss))

        epoch_losses.append(epoch_loss)

    epoch_losses = np.array(epoch_losses)
    np.save("./generator_losses.npy", epoch_losses)

    torch.save(model.state_dict(), "./pretrained_generator_model.pth")

    return model

def generator_labels(trainLoader: DataLoader, model: AE, arch_size: int = 1024):
    """
    Generators labels from the pretrained generator model
    """

    labels_g = [0] * len(trainLoader.dataset) # Label for each data point
    bat_size = trainLoader.dataset
    arch_size = arch_size # Size of Architecture

    with torch.no_grad():
        for i, data in tqdm(enumerate(trainLoader)):
            losses = []
            data = data.to(device)

            reconstructed = model.forward(data)

            # Loss function
            for dataBatch in range(data.shape[0]):

                j1 = data[dataBatch].cpu().detach().numpy().reshape((arch_size,))
                r1 = reconstructed[dataBatch].cpu().detach().numpy().reshape((arch_size,))
                diff = np.subtract(j1, r1)

                l2_loss = np.linalg.norm(diff)

                losses.append(l2_loss)

            losses = np.array(losses)
            sr = np.std(losses)
            ur = np.mean(losses)
            th = ur + sr # Calculating threshold as mentioned in paper

            for loss in range(0, len(losses)):

                if losses[loss] >= th:
                    lables_g [ i * bat_size + loss] = 1

            labels_g = [np.array([i], dtype = np.float32).reshape(1) for i in labels_g]

    labels_g = torch.tensor(np.array(labels_g)).to(device)

    return labels_g


def train_discriminator(trainLoader: DataLoader, labels_g, model: Discriminator, optimizer: Optimizer ,lossFunctionD, epochs: int):
    """
    Training loop for discriminator
    """
    losses = []
    epoch_losses = []

    batchSize = trainLoader.dataset

    for epoch in range(epochs):

        running_loss = 0.0
        print("Epoch No.: {}".format(epoch))

        for i, data in tqdm(enumerate(trainLoader)):

            data = data.to(device)
            output = model.forward(data).reshape(data.shape[0],)

            loss = lossFunctionD(output,labels_g[i*batchSize : (i*batchSize + data.shape[0])].reshape(data.shape[0],))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(trainLoader)
        print("Loss: {}".format(epoch_loss))

        epoch_losses.append(epoch_loss)

    epoch_losses = np.array(epoch_losses)

    np.save("./discriminator_losses.npy", epoch_losses)
    torch.save(model.state_dict(), "./pretrained_discriminator_model.pth")


def discriminator_labels(trainLoader: DataLoader, model: Discriminator):
    """
    Generates labels from discriminator
    """

    labels_d = [0] * len(trainLoader.dataset)

    with torch.no_grad():

        for i, data in tqdm(enumerate(trainLoader)):

            data = data.to(device)
            output = model.forward(data).reshape(data.shape[0],)

            output = output.cpu().detach().numpy()
            sr = np.std(output)
            ur = np.mean(output)

            th = ur + (0.1) * sr

            for out in range(len(output)):
                if output[out] >= th:
                    labels_d[i * bat_size + out] = 1

            labels_d = [np.array([label],dtype=np.float32).reshape(1) for label in labels_d]

    labels_d = np.array(labels_d)
    np.save('./pretrained_discriminator_labels.npy', labels_d)


