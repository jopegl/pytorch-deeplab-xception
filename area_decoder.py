import torch
import torch.nn as nn
import torch.nn.functional as F

class AreaDecoder(nn.Module):
    def __init__(self, high_in_features, low_in_features, out_channels, ):
        super().__init__()
        self.total_in = high_in_features + low_in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.total_in, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ),
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ),

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_level_features, high_level_features, original_image_size):
        x = torch.cat([low_level_features, high_level_features], dim = 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.interpolate(x,
            size = original_image_size,
            mode = 'bilinear',
            align_corners = True
        )
        return x
