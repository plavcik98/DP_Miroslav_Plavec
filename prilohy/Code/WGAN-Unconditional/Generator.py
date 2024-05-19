import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels, image_channels=1, features = [512, 256, 128, 64, 32, 16, 8]):
        super(Generator, self).__init__()

        self.model = nn.ModuleList()
        self.model.append(self.block(in_channels, features[0], (4,2), 1, 0))
        in_channels = features[0]
        for feature in features[1:]:
            self.model.append(self.block(in_channels, feature, 4, 2, 1))
            in_channels = feature
        
        self.model.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels, image_channels, 4, 2, 1),
            nn.Tanh()
        ))


    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    
    def forward(self, x):
        for block in self.model:
            x = block(x)
        
        return x