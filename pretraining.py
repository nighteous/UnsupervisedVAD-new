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
# print(num_output_features)

# feats = []
# for k in range(1, 1 + num_output_features):
#   z=np.load(path+'output'+str(k)+'.npy')
#   z=np.array(z,dtype=np.float32)
#   feats.append(z)

x = os.listdir(path)
feats = []

# for i in x:
#     z = os.listdir(os.path.join(path, i))
#     for j in z:
#         feat_npy = np.load(os.path.join(os.path.join(path, i), j))
#         feat_npy = np.array(feat_npy, dtype = np.float32)
#         feats.append(feat_npy)

for i in x:
    z = os.listdir(os.path.join(path, i))
    for j in z:
        feat_npy = np.load(os.path.join(os.path.join(path, i), j))
        feat_npy = np.array(feat_npy, dtype = np.float32)

        for k in feat_npy:
            feats.append(k)

del feat_npy # to delete feat_npy because we dont need it anymore

print("Shape of features is {}".format(np.array(feats).shape))

# # Making clean dataset for pre-training purpose.
# clean_feat = []

# for t in range(len(feats) - 1):
#   diff = feats[t+1] - feats[t]
#   value = np.linalg.norm(diff)
#   if(value <= 0.7):
#     clean_feat.append(feats[t+1])

# clean_feat = np.array(clean_feat)
# np.save('./clean_feat', clean_feat)

clean_feat = np.array(feats) # with i3d no feature difference is below given Dth threshold so removed 

# print("Shape of cleaned features is {}".format(clean_feat.shape))
# exit()
train_dataset=DataLoader(clean_feat,batch_size=8192,shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 200

# Model dec
netG = AE()
netG.to(device)

# reconstruction loss (MSE)
loss_function_G = torch.nn.MSELoss()

#RMSprop optimizer with a learning rate of 0.00002,
#momentum 0.60, for 15 epochs on training data with batch size 8192
optimizer = RMSprop(netG.parameters(), lr=0.00002, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.6, centered=False)

outputs = []
losses = []
epoch_losses = []
for epoch in range(epochs):

    running_loss = 0.
    print("Epoch No.: {}".format(epoch+1))

    #for (image, inputs) in tqdm(train_dataset):
    for i,data in tqdm(enumerate(train_dataset)):
        # Output of Autoencoder
        data = data.to(device)

        reconstructed = netG.forward(data)

        # Calculating the loss function
        loss = loss_function_G(reconstructed, data)
        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Storing the losses in a list for plotting
        losses.append(loss.item())
        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(train_dataset)
    # epoch_acc = running_corrects / len(normal_train_dataset) * 100.
    print("Loss: {}".format(epoch_loss))
    epoch_losses.append(epoch_loss) 
    outputs.append((epochs, i, reconstructed))

epoch_losses = np.array(epoch_losses)

np.save("./Pre_Training/generator_losses.npy", epoch_losses)
path='./Pre_Training/pretrained_generator_model.pth'
torch.save(netG.state_dict(),path)

# print("BLOCK    ", 1)

labels_g=[0]*(len(train_dataset.dataset)) # we need to calculate label for each data point
bat_size = train_dataset.batch_size
arch_size = 1024

with torch.no_grad():
    for i,data in tqdm(enumerate(train_dataset)):
      losses=[]
      data = data.to(device)
      # Output of Autoencoder
      reconstructed = netG.forward(data)
      # Calculating the loss function
      for data_bat in range(data.shape[0]):
        j1=data[data_bat].cpu().detach().numpy().reshape((arch_size,))
        r1=reconstructed[data_bat].cpu().detach().numpy().reshape((arch_size,))
        diff = np.subtract(j1, r1)
        l2_loss=np.linalg.norm(diff)
        losses.append(l2_loss)

      losses=np.array(losses)
      sr=np.std(losses)
      ur=np.mean(losses)
      th=ur+sr
      for loss in range(0,len(losses)):
        if(losses[loss]>=th):
          labels_g[i*bat_size + loss]=1
      labels_g = [np.array([i],dtype=np.float32).reshape(1) for i in labels_g]

labels_g = torch.tensor(labels_g)
labels_g = labels_g.to(device)


# print("BLOCK   ", 2)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model dec
netD = Discriminator()
netD.to(device)

# reconstruction loss (MSE)
loss_function_D = torch.nn.BCELoss() # I think we should you bcewithlogitsloss.

#RMSprop optimizer with a learning rate of 0.00002,
#momentum 0.60, for 15 epochs on training data with batch size 8192
optimizer = RMSprop(netD.parameters(), lr=0.00002, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.6, centered=False)

outputs = []
losses = []
epoch_losses = []

disc_prob=[]*len(train_dataset.dataset)
#disc_prob=torch.tensor(disc_prob)
bat_size = train_dataset.batch_size
for epoch in range(epochs):

    running_loss = 0.
    print("Epoch No.: {}".format(epoch))

    for i,data in tqdm(enumerate(train_dataset)):
        data = data.to(device)
        output = netD.forward(data).reshape(data.shape[0],)
        # Calculating the loss function
        loss = loss_function_D(output,labels_g[i*bat_size : (i*bat_size + data.shape[0])].reshape(data.shape[0],)) # Here basically we need to move the labels like 0-5, 5-10, 10-12.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Storing the losses in a list for plotting
        losses.append(loss.item())
        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(train_dataset)
    # epoch_acc = running_corrects / len(normal_train_dataset) * 100.
    print("Loss: {}".format(epoch_loss))
    epoch_losses.append(epoch_loss)
    outputs.append((epochs, i, reconstructed))

epoch_losses = np.array(epoch_losses)

np.save("./discriminstor_losses.npy", epoch_losses)
path='./pretrained_discriminator_model.pth'
torch.save(netD.state_dict(), path)



# print("BLOCK   ", 3)



labels_d=[0]*(len(train_dataset.dataset))

with torch.no_grad():
    for i,data in tqdm(enumerate(train_dataset)):
        data = data.to(device)
        output = netD.forward(data).reshape(data.shape[0],)

        # for data_bat in range(data.shape[0]):
        #   # print(output[data_bat])
        #   # print(labels[i*bat_size + data_bat])
        #   loss = loss_function_D(output[data_bat],labels_g[i*bat_size + data_bat].reshape(()))
        #   losses_d.append(loss.item())

        output = output.cpu().detach().numpy()
        sr=np.std(output)
        ur=np.mean(output)
        th=ur+(0.1)*sr

        for out in range(0,len(output)):
          if(output[out]>=th):
            labels_d[i*bat_size + out]=1
        labels_d = [np.array([label],dtype=np.float32).reshape(1) for label in labels_d]

labels_d=np.array(labels_d)
np.save('./pretrained_discriminator_labels', labels_d)




# print("BLOCK   ", 4)

