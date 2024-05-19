import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels, features=[8,16,32,64,128,256,500]):
        super(Generator, self).__init__()

        # Down part
        self.downs = nn.ModuleList()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        in_channels = features[0]
        for feature in features[1:]:
            self.downs.append(self._down(in_channels, feature))
            in_channels = feature
        
        self.bottlneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(4,2), stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2))
                

        # UP part
        self.ups = nn.ModuleList()
        
        self.firstUp = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(4,2), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())

        
        features = features[::-1]
        in_channels = features[0]

        for feature in features[1:]:
            self.ups.append(self._up(in_channels*2, feature))
            in_channels = feature
            

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels*2, 1, kernel_size=4, stride=2, padding=1),
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
        x = self.initial_down(x)
        skip = [x]
        for down in self.downs:
            x = down(x)
            skip.append(x)

        x = self.bottlneck(x)
        skip = skip[::-1]
        x = x * z
        
        x = self.firstUp(x)
        
        for up, s in zip(self.ups, skip[:-1]):
            x = up(torch.cat([x, s], dim=1))

        x = self.final_conv(torch.cat([x, skip[-1]], dim=1))

        return x