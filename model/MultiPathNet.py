"""MultiPathNet: single model file, no cross-import from other model modules."""
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


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UpBlock_RefStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock_RefStack, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')
        self.conv2 = nn.Conv2d(3 * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip1, skip2], dim=1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class MultiPathNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiPathNet, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down_ori1 = DownBlock(in_channels, filters[0])
        self.down_ori2 = DownBlock(filters[0], filters[1])
        self.down_ori3 = DownBlock(filters[1], filters[2])
        self.down_ori4 = DownBlock(filters[2], filters[3])
        self.down_mig1 = DownBlock(in_channels, filters[0])
        self.down_mig2 = DownBlock(filters[0], filters[1])
        self.down_mig3 = DownBlock(filters[1], filters[2])
        self.down_mig4 = DownBlock(filters[2], filters[3])
        self.bottleneck = Bottleneck(filters[3] * 2, filters[4])
        self.up1 = UpBlock_RefStack(filters[4], filters[3])
        self.up2 = UpBlock_RefStack(filters[3], filters[2])
        self.up3 = UpBlock_RefStack(filters[2], filters[1])
        self.up4 = UpBlock_RefStack(filters[1], filters[0])
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x_ori, x_mig):
        skip_ori1, p_ori1 = self.down_ori1(x_ori)
        skip_ori2, p_ori2 = self.down_ori2(p_ori1)
        skip_ori3, p_ori3 = self.down_ori3(p_ori2)
        skip_ori4, p_ori4 = self.down_ori4(p_ori3)
        skip_mig1, p_mig1 = self.down_mig1(x_mig)
        skip_mig2, p_mig2 = self.down_mig2(p_mig1)
        skip_mig3, p_mig3 = self.down_mig3(p_mig2)
        skip_mig4, p_mig4 = self.down_mig4(p_mig3)
        p4 = torch.cat([p_ori4, p_mig4], dim=1)
        bn = self.bottleneck(p4)
        u1 = self.up1(bn, skip_mig4, skip_ori4)
        u2 = self.up2(u1, skip_mig3, skip_ori3)
        u3 = self.up3(u2, skip_mig2, skip_ori2)
        u4 = self.up4(u3, skip_mig1, skip_ori1)
        outputs = F.relu(self.final_conv(u4))
        return outputs
