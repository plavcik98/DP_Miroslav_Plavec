from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms
from torchvision.io import read_image
import random
import pandas as pd

class LungsDataset(Dataset):
    def __init__(self, csv_path):
        self.csv = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.csv)
    
    def transform(self, image, mask):
        resize = transforms.Resize(size=(512, 256), antialias=True)
        image = resize(image)
        mask = resize(mask)

        norm = transforms.Normalize((0.5), (0.5))
        image = norm(image)
        mask = norm(mask)
        
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        

        return image, mask
    
    def __getitem__(self, index):

        frame_path = self.csv.iloc[index][0]
        mask_path = self.csv.iloc[index][1]

        frame = read_image(frame_path, torchvision.io.ImageReadMode.GRAY).type(torch.FloatTensor)  / 255
        mask = read_image(mask_path, torchvision.io.ImageReadMode.GRAY).type(torch.FloatTensor)  / 255

        frame, mask = self.transform(frame, mask)

        sample = {0: frame, 1: mask}

        return sample
    
    def getEvalMask(self, n):
        selected_masks = self.csv.sample(n)
        masks = []
        idxs = []
        for idx, row in selected_masks.iterrows():
            mask = self.__getitem__(idx)[1]
            masks.append(mask)
            idxs.append(idx)
        
        self.csv.drop(index=idxs)
        # self.csv = self.csv.reset_index(drop=True)
        return torch.stack(masks)