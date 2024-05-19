import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, features=[8,16,32,64,128,256,500]):
        super(Generator, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Down 
        for feature in features:
            self.downs.append(self._down(in_channels, feature))
            in_channels = feature
        
        self.bottlneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], kernel_size=(4,2), stride=1, padding=0),
            nn.BatchNorm2d(features[-1]),
            nn.LeakyReLU(0.2))
        
        features = features[::-1]
        
        self.firstUp = nn.Sequential(
            nn.ConvTranspose2d(features[0], features[0], kernel_size=(4,2), stride=1, padding=0),
            nn.BatchNorm2d(features[0]),
            nn.ReLU())

        # UP
        for feature in features[1:]:
            self.ups.append(self._up(in_channels, feature))
            in_channels = feature
            

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(features[-1], 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def _down(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))
    
    def _up(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    
    def forward(self, x, z):
        for down in self.downs:
            x = down(x)

        x = self.bottlneck(x)
        
        x = x * z
        
        x = self.firstUp(x)
        
        for up in self.ups:
            x = up(x)

        x = self.final_conv(x)
        
        return x