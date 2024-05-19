import torch
import torch.nn as nn

class Critic(nn.Module):

    def __init__(self, in_channels, features = [8,16,32,64,128,256,500]):
        super(Critic, self).__init__()
        
        self.downs = nn.ModuleList()

        for feature in features:
            self.downs.append(self._down(in_channels, feature))
            in_channels = feature
        

        self.final_conv = nn.Sequential(
            nn.Conv2d(features[-1], 1, kernel_size=(4,2), stride=1, padding=0)
        )
    

    def _down(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2))


    def forward(self, x, mask):
        x = torch.concat((x, mask), dim=1)

        for down in self.downs:
            x = down(x)
        
        x = self.final_conv(x)
        
        return x