import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels, features = [8, 16, 32, 64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.model = nn.ModuleList()
        for feature in features:
            self.model.append(self.block(in_channels, feature, 4, 2, 1))
            in_channels = feature
        
        # final conv
        self.model.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 1, (4, 2), 2, 0),
                nn.Sigmoid()
                ))
    

    def block(self, in_channels, out_channels, kerne_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kerne_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):

        for block in self.model:
            x = block(x)
            # print(x.shape)
        
        return x