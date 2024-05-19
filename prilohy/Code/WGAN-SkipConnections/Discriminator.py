import torch
import torch.nn as nn

class Critic(nn.Module):

    def __init__(self, in_channels, features = [8,16,32,64,128,256,500]):
        super(Critic, self).__init__()
        
        self.downs = nn.ModuleList()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        in_channels = features[0]
        for feature in features[1:]:
            self.downs.append(self._down(in_channels, feature))
            in_channels = feature
        

        self.final_conv = nn.Sequential(
            nn.Conv2d(features[-1], 1, kernel_size=(4,2), stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
    

    def _down(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2))


    def forward(self, x, mask):
        x = torch.concat((x, mask), dim=1)
        
        x = self.initial_down(x)
        #print(x.shape)
        for down in self.downs:
            x = down(x)
            #print(x.shape)
        
        x = self.final_conv(x)
        #print(x.shape)
        
        return x