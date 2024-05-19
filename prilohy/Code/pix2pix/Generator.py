import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down, activation, use_dropout):
        super(Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation=="relu" else nn.LeakyReLU(0.2)
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x) if self.use_dropout else x

        return x


class Generator(nn.Module):
    def __init__(self, in_channels, features = [8, 16, 32, 64, 128, 256, 512]):
        super(Generator, self).__init__()
        
        self.in_channels = in_channels

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.BatchNorm2d(features[0]),
            nn.LeakyReLU(0.2)
        )

        # donw part
        self.downs = nn.ModuleList()
        in_channels = features[0]
        for feature in features[1:]:
            self.downs.append(
                Block(in_channels, feature, down=True, activation="leaky", use_dropout=False)
            )
            in_channels = feature
        
        # bottlneck
        self.bottlneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (4, 2), 1, 0, padding_mode="reflect"),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2)
        )

        self.first_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, (4, 2), 1, 0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # up part
        self.ups = nn.ModuleList()
        features = features[::-1]
        in_channels = features[0]
        n = 0
        for feature in features[1:]:
            self.ups.append(
                Block(in_channels*2, feature, down=False, activation="relu", use_dropout=True if n<3 else False)
            )
            in_channels = feature
            n += 1
        
        # final conv
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels*2, self.in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    
    def forward(self, mask):
        x = self.initial_down(mask)
        skip = [x]
        for down in self.downs:
            x = down(x)
            skip.append(x)
        
        x = self.bottlneck(x)
        skip = skip[::-1]
        
        x = self.first_up(x)

        for up, s in zip(self.ups, skip[:-1]):
            x = up(torch.cat([x, s], dim=1))
        

        x = self.final_conv(torch.cat([x, skip[-1]], dim=1))

        return x


        




