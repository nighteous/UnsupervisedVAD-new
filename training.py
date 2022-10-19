import sys
from Architecture.generator import *
from Architecture.discriminator import *
import os
import numpy as np
import torch
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loading the pre-trained discriminator model
netD = Discriminator()
netD.load_state_dict(torch.load('/shared/home/v_varenyam_bhardwaj/local_scratch/UnsupervisedVAD/UnsupervisedVAD/pretrained_discriminator_model.pth'))
netD = netD.to(device)
netD.eval()

# loading the pre-trained generator model
netG = AE()
netG.load_state_dict(torch.load('/shared/home/v_varenyam_bhardwaj/local_scratch/UnsupervisedVAD/UnsupervisedVAD/Pre_Training/pretrained_generator_model.pth'))
netG = netG.to(device)
netG.eval()

# print(netD)
# print(netG)

netD = netD.to(device)
loss_function_D = torch.nn.BCELoss() # D
loss_function_G = torch.nn.MSELoss() # G

#RMSprop optimizer with a learning rate of 0.00002,
#momentum 0.60, for 15 epochs on training data with batch size 8192
optimizer = RMSprop(netD.parameters(), lr=0.00002, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.6, centered=False)

path = '/shared/home/v_varenyam_bhardwaj/local_scratch/Dataset/UCF-Crime/all_rgbs'
num_output_features = len(os.listdir(path))
# print(num_output_features)

x = os.listdir(path)
feats = []
for i in x:
    z = os.listdir(os.path.join(path, i))
    for j in z:
        feat_npy = np.load(os.path.join(os.path.join(path, i), j))
        feat_npy = np.array(feat_npy, dtype = np.float32)

        for k in feat_npy:
            feats.append(k)

# del feat_npy # to delete feat_npy because we dont need it anymore

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

clean_feat = np.array(feats)

print(clean_feat.shape)
# exit()
train_dataset=DataLoader(clean_feat,batch_size=8192,shuffle=True) # original 8192

epochs = 200
outputs = []
losses_G_whole = [] # what is the loss for each epoch
losses_D_whole = [] # what is the loss for each epoch
Losses= []
bat_size = train_dataset.batch_size
num_batches = len(train_dataset)
num_samples = len(train_dataset.dataset)
labels_d=np.load('pretrained_discriminator_labels.npy')

for epoch in range(epochs):

    losses_G = [] # what is the loss for each sample, used for psuedo labels generation.
    losses_D = [] # what is the loss for each sample, used for psuedo labels generation.
    output_batch = [[]]*num_batches

    running_loss_G = 0.
    running_loss_D = 0
    print("Epoch No.: {}".format(epoch))

    # Training of the autoencoder
    for idx,data in tqdm(enumerate(train_dataset)):
        loss_batch = 0
        data = data.to(device)
        # Output of Autoencoder
        reconstructed = netG.forward(data)
        optimizer.zero_grad()
        loss_temp = loss_function_G(data, reconstructed)
        # label = labels_d[idx]

        # calculating loss for every data point in a batch
        for data_point in range(data.shape[0]):
          label = labels_d[idx*bat_size + data_point]

          # Negative Learning
          if(label == 0):
            loss = loss_function_G(reconstructed[data_point], data[data_point])
            
          else:
            loss = loss_function_G(reconstructed[data_point], torch.ones(1024).reshape((1,1,-1)).to(device))
          
          losses_G.append(loss.item())
          loss_batch += loss.item()
        
        loss_batch = loss_batch/bat_size
        loss_temp.data = torch.Tensor([loss_batch])
        loss_temp.backward()

        optimizer.step()
        running_loss_G += loss_batch * data.size(0) # running_loss_G is the sum of losses for all the 20 feature vectors 
    losses_G_whole.append(running_loss_G)  


    # Pseudo labels generated from the autoencoder
    labels_g=[0]*(num_samples) # labels will be generated for each data point and not whole train_dataset.
    # losses=[]
    # with torch.no_grad():
    #   for i,j in tqdm(enumerate(train_dataset)):
    #   # Output of Autoencoder
    #     reconstructed = netG.forward(j)
    #   # Calculating the loss function
    #   #loss = loss_function(reconstructed, j)
    #     # print(j.shape)
    #     j1=j.cpu().detach().numpy().reshape((2048,))
    #     r1=reconstructed.cpu().detach().numpy().reshape((2048,))
    #     x=[0]*(len(j1))
    #     for i in range(0,len(j1)):
    #       x[i]=j1[i]-r1[i]
    #     l=np.linalg.norm(x)
    #     losses.append(l)

    losses_G=np.array(losses_G)

    for batch in range(num_batches):
      idx1 = batch*bat_size
      idx2 = idx1 + bat_size
      if(idx2 > len(losses_G)):
        idx2 = len(losses_G)
      losses_batch = losses_G[idx1 : idx2] #
      sr=np.std(losses_batch)
      ur=np.mean(losses_batch)
      th=ur+sr
      for loss in range(0,len(losses_batch)):
        if(losses_G[loss]>=th):
          labels_g[idx1 + loss]=1
    
    labels_g = [np.array([label],dtype=np.float32).reshape(1) for label in labels_g]
    labels_g = torch.tensor(labels_g)
    labels_g = labels_g.to(device)

    # Training of the discriminator
    for idx,data in tqdm(enumerate(train_dataset)):
      data = data.to(device)
        # output of discriminator
      output = netD.forward(data).reshape(data.shape[0],)

      loss_D = loss_function_D(output, labels_g[idx*bat_size : (idx*bat_size + data.shape[0])].reshape(data.shape[0], ))
        
      optimizer.zero_grad()
      loss_D.backward()
      optimizer.step()
        
      losses_D.append(loss_D.item())
      running_loss_D += loss_D.item() * data.size(0)

      output_batch[idx] = output.cpu().detach().numpy()
    losses_D_whole.append(running_loss_D)


    # Psuedo label generation from discriminator
    labels_d=[0]*(num_samples)
    # losses_d=[]
    # with torch.no_grad():
    #   for i,j in tqdm(enumerate(train_dataset)):
    #     output = netD.forward(j).reshape(1,)
    #     loss = loss_function(output,torch.tensor(labels[i]))
    #     losses_d.append(loss)

    for batch in range(num_batches):
      output_per_batch = output_batch[batch]
      sr=np.std(output_per_batch)
      ur=np.mean(output_per_batch)
      th=ur+(0.1)*sr

      for output in range(0,len(output_per_batch)):
        if(output_per_batch[output]>=th):
          labels_d[batch*bat_size + output]=1

    # losses_D=np.array(losses_D)
    # sr=np.std(losses_D)
    # ur=np.mean(losses_D)
    # th=ur+(0.1)*sr
    # for loss in range(0,len(losses_D)):
    #   if(losses_D[loss]>=th):
    #     labels_d[loss]=1
    labels_d = [np.array([label],dtype=np.float32).reshape(1) for label in labels_d]
   

    epoch_loss = running_loss_D / len(train_dataset)
    Losses.append(epoch_loss)
    # epoch_acc = running_corrects / len(normal_train_dataset) * 100.
    print("Loss: {}".format(epoch_loss))
    outputs.append((epochs, idx, reconstructed))

np.save("./Losses.npy", np.array(Losses))
np.save("./losses_G.npy", losses_G)
np.save("./losses_D.npy", losses_D_whole)
