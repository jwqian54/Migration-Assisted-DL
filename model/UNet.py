"""UNet: single model file, no cross-import from other model modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        p = self.pool(x)
        return x, p


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])
        self.bottleneck = Bottleneck(filters[3], filters[4])
        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)
        bn = self.bottleneck(p4)
        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        outputs = F.relu(self.final_conv(u4))
        return outputs
