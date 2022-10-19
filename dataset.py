import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset

from PIL import Image
import os
import cv2

# Instead of having the segmeneter to divide the dataset into segments
# I'll let the dataset module to handle it (makes it easier to load the dataset as well)


class Dataset(Dataset):
    """
    Custom Dataset for loading all images in Dataset Folder after segmentation.
    """

    def __init__(self, path) -> None:
        super().__init__()

        self.path = path

        self.listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(self.path):
            print(dirnames)
            self.listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        self.transfrom = transforms.Compose([
            transforms.CenterCrop((256,256)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.listOfFiles)

    def __getitem__(self, index):

        image = cv2.imread(self.listOfFiles[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        image = self.transfrom(image)

        return image


