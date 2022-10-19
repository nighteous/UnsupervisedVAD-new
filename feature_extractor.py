# Check if code is runninng in colab or not
import sys
sys.path.insert(1, '/shared/home/v_varenyam_bhardwaj/local_scratch/UnsupervisedVAD/UnsupervisedVAD/')
from ast import Pass
import numpy as np
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False


# in this block, frames are transformed and turned into a VideoFrameDataset
import numpy as np
# import matplotlib.pyplot as plt

if IN_COLAB:
    from UnsupervisedVAD.video_dataset import VideoFrameDataset, ImglistToTensor
else:
    from video_dataset import VideoFrameDataset, ImglistToTensor

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

if IN_COLAB:
    path = './UnsupervisedVAD/'

else:
    path = './'
import numpy as np
import matplotlib.pyplot as plt
from UnsupervisedVAD.video_dataset import VideoFrameDataset, ImglistToTensor
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import cv2
import numpy as np
import os
import json

path = './UnsupervisedVAD/Dataset/Frames/'

transfrom = transforms.Compose([
            ImglistToTensor(),
            transforms.CenterCrop((256,256)),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
import os
# print(os.system("pwd"))

if IN_COLAB:
    file = open('./UnsupervisedVAD/train.txt', 'r')
else:
    file = open('./train.txt', 'r')

# data = file.readlines()
# lines = [line.split() for line in data]

# # to find least number of frames
# frames = [line[2] for line in lines]
# minFrame = min(frames)

dataset = VideoFrameDataset(path + 'Dataset/Frames/', path + 'train.txt', num_segments=1, frames_per_segment=16,
                            imagefile_template='frame_{:05d}.jpg', transform=None, test_mode=False)
# torch.save(dataset, './dataset1')
print(len(dataset))
# num_anomaly = os.listdir('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dataset/Frames')

# total = 0
# for i in num_anomaly:
#     total+=len(os.listdir('/shared/home/v_rahul_pratap_singh/local_scratch/UnsupervisedVAD/Dataset/Frames/'+i))
# print(len(num_anomaly))
# exit(len(total))

# dataset = VideoFrameDataset(path + 'Dataset/Frames/', path + 'train.txt', num_segments=int(minFrame)//16, frames_per_segment=16,
#                             imagefile_template='frame_{:05d}.jpg', transform=None, test_mode=False)


# Segments formation
segments = [[]] * len(dataset) # Get empty list of empty lists for each video in dataset
from tqdm import tqdm
import os
for j in range(len(dataset)):
    print("Video {}".format(j))
    sample = dataset[j]

    frames = sample[0]

    segment = []

    for i in tqdm(range(len(frames)//16)):
        #print("Segment {}".format(i))
        segment_i = frames[16 * i: 16 * (i + 1)]
        segment.append(segment_i)

    segments[j] = segment

print("Segmenting done")

segments = np.array(segments)


import cv2
import numpy as np

# This is to run the code without any interuptions
if IN_COLAB:
    from google.colab.patches import cv2_imshow as cv2_imshow
else:
    from cv2 import imshow as cv2_imshow

frameSize = (224, 224)
vid_tensors=[[]] * len(dataset)

#vid_tensors[j]= list of 10 tensors for the 10 segments of video no. j

dataset = VideoFrameDataset(path, './UnsupervisedVAD/train.txt', num_segments=100, frames_per_segment=16, imagefile_template='frame_{:05d}.jpg', transform=None, test_mode=False)
print("Dataset Formed")

segments = [[]] * len(dataset) # Get empty list of empty lists for each video in dataset
for j in range(len(dataset)):

    sample = dataset[j]
    frames = sample[0]
    # print(len(frames))

    segment = []

    for i in range(len(frames)//16):
        segment_i = frames[16 * i: 16 * (i + 1)]
        segment.append(segment_i)


    segments[j] = segment

segments = np.array(segments)
print("Segmenting done")


frameSize = (224, 224)
vid_tensors=[[]] * len(dataset)

#vid_tensors[j]= list of  tensors for the segments of video no. j






for i in range(len(segments)): #get_video
    lt=[]
    #print('video no.',i)
    vid=segments[i]
    frameset=np.array(vid) #array of 10 segments
    #print('no. of segments in video ',i,': ',len(frameset))
    for seg_index in range(0,len(frameset)):
        #print('seg_index',seg_index)
    frameset=np.array(vid) #array of segments
    for seg_index in range(0,len(frameset)):
    frameset=np.array(vid) #array of segments
    for seg_index in range(0,len(frameset)):
        seg=frameset[seg_index]
        l=[]
        for k in range(0,len(seg)): #16 frames per segment
            #print('frame no.',k)
            pil_img=seg[k]
            cv_img=np.array(pil_img)
            cv_img=cv2.resize(cv_img,(112,112))
            #cv2_imshow(cv_img)
            #print(cv_img.shape)
            l.append(cv_img)

        t=tuple(l)

        x = np.stack(t, axis = -1)
        y= np.transpose(x, (3,2,1,0)) #tensor for one segment
        #print('tensor for segment no. ', seg_index)
        #print('appending tensor for this segment')
        #print('vdi',vid_tensors[i])
        lt.append(y)
        #print('current shape',np.array(vid_tensors[0]).shape,np.array(vid_tensors[1]).shape)
    vid_tensors[i]=lt

    #print(i,len(vid_tensors[i]))

#cv2.destroyAllWindows()
vid_tensors=np.array(vid_tensors)
        t=tuple(l)
        
        x = np.stack(t, axis = -1)
        y= np.transpose(x, (3,2,1,0)) #tensor for one segment
        lt.append(y)
    vid_tensors[i]=lt


vid_tensors=np.array(vid_tensors)
print("Converted into required format")

data_segments=[]

for j in range(0,len(vid_tensors)):
    for k in range(0,len(vid_tensors[j])):
        data_segments.append(vid_tensors[j][k])
print(len(data_segments))

# Create json file for feature extractor
import os
import json
print("Added all segments together")
print(len(data_segments))

print("Added all segments together")
print(len(data_segments))


video_path = '/content/outpy.avi'
output_file = '/content/outpy.npy'
count=0
ls=[]
for s in range(0, len(data_segments)):
    count+=1
    tens=data_segments[j]
    op_d={'video': tens, 'input': video_path, 'output': output_file}
    op_d_new=op_d
    op_d_new['video']=np.array(op_d['video']).tolist()
    ls.append(op_d_new)
    print('writing line ',count)
json_object=json.dumps(ls)
with open("sample.json", "w") as outfile:
    outfile.write(json_object)
<<<<<<< HEAD
<<<<<<< HEAD

print("Done making json")

# Create folder for output of features
import os
if not os.path.exists("./output_features/"):
    os.mkdir('./output_features/')


os.system('python ./video_feature_extractor/extract.py --jsn="./sample.json" --type=3d --batch_size=1 --resnext101_model_path=./resnext101.pth')
=======
=======
>>>>>>> db11177aab58bf98c00b70e9c3de846f895c69b4
print("Converted to json")    

if not os.path.exists('./output_features/'):
  os.mkdir('./output_features/')
  
os.system('python ./video_feature_extractor/extract.py --jsn="sample.json" --type=3d --batch_size=1 --resnext101_model_path=/content/resnext101.pth') # basically running extract.py

<<<<<<< HEAD
>>>>>>> db11177aab58bf98c00b70e9c3de846f895c69b4
=======
>>>>>>> db11177aab58bf98c00b70e9c3de846f895c69b4
