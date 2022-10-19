import matplotlib.pyplot as plt
from video_dataset import VideoFrameDataset

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


path = './Dataset/Frames/'

transform = transforms.Compose([transforms.ToTensor])

dataset = VideoFrameDataset(path, './train.txt', num_segments=1000, frames_per_segment=16, imagefile_template='frame_{:05d}.jpg', transform=transform, test_mode=False)

dataloader = DataLoader(dataset, batch_size=10)

print(len(list(enumerate(dataloader))))
