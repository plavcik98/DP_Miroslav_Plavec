import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        x = self.block(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, features = [64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        self.model = nn.ModuleList()

        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(features[0]),
            nn.LeakyReLU(0.2)
        )

        self.model.append(self.initial_layer)

        # layers = []
        in_channels = features[0]
        for feature in features[1:]:
            self.model.append(
                CNNBlock(in_channels, feature, 4, stride = 1 if feature==features[-1] else 2, padding=1))
            in_channels = feature
        
        self.model.append(
            nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.Sigmoid())
        )
        # self.model = nn.Sequential(*layers)
    
    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)

       # x = self.initial_layer(x)

        x = self.model(x)


        return x


        

        
