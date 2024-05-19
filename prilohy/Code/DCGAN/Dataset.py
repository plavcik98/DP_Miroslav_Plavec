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
    
    def transform(self, image):
        resize = transforms.Resize(size=(512, 256), antialias=True)
        image = resize(image)

        norm = transforms.Normalize((0.5), (0.5))
        image = norm(image)
        
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)     

        return image
    
    def __getitem__(self, index):

        frame_path = self.csv.iloc[index][0]

        frame = read_image(frame_path, torchvision.io.ImageReadMode.GRAY).type(torch.FloatTensor)  / 255

        frame = self.transform(frame)

        return frame
