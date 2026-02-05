"""DPENet: single model file, no cross-import from other model modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ResBlcok_wo_downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_wo_downsample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + x)
        return out


class ResBlcok_w_downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_w_downsample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        x = self.conv1_1(x)
        out = F.relu(out + x)
        return out


class DUBV2(nn.Module):
    def __init__(self, in_channels, out_channels, H):
        super(DUBV2, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.pool = nn.AdaptiveMaxPool2d((H, H))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.pool(x)
        return x


class DPENet(nn.Module):
    def __init__(self, in_channels=1, out_channels=None, atrous_rates=None):
        super(DPENet, self).__init__()
        if out_channels is None:
            out_channels = [64, 128, 256, 512]
        if atrous_rates is None:
            atrous_rates = [6, 12, 18]
        self.preconv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], 7, 2, padding=3),
            nn.BatchNorm2d(out_channels[0]), nn.ReLU(), nn.MaxPool2d(3, 2, 1)
        )
        self.resblock1 = nn.Sequential(
            ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
            ResBlcok_wo_downsample(out_channels[0], out_channels[0])
        )
        self.resblock2 = nn.Sequential(
            ResBlcok_w_downsample(out_channels[0], out_channels[1]),
            ResBlcok_wo_downsample(out_channels[1], out_channels[1])
        )
        self.resblock3 = nn.Sequential(
            ResBlcok_w_downsample(out_channels[1], out_channels[2]),
            ResBlcok_wo_downsample(out_channels[2], out_channels[2])
        )
        self.resblock4 = nn.Sequential(
            ResBlcok_w_downsample(out_channels[2], out_channels[3]),
            ResBlcok_wo_downsample(out_channels[3], out_channels[3])
        )
        self.dub1 = DUBV2(out_channels[0], out_channels[0], 4)
        self.dub2 = DUBV2(out_channels[1], out_channels[1], 4)
        self.dub3 = DUBV2(out_channels[2], out_channels[2], 4)
        self.dub4 = DUBV2(out_channels[3], out_channels[3], 4)
        self.cbam = CBAM(sum(out_channels))
        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3], out_features=2), nn.ReLU())
        self.postconv = nn.Sequential(
            nn.Conv2d(sum(out_channels), out_channels[3], 1, 1),
            nn.BatchNorm2d(out_channels[3]), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.preconv1(x)
        x1 = self.resblock1(x)
        x2 = self.resblock2(x1)
        x3 = self.resblock3(x2)
        x4 = self.resblock4(x3)
        x1 = self.dub1(x1)
        x2 = self.dub2(x2)
        x3 = self.dub3(x3)
        x4 = self.dub4(x4)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.cbam(out)
        out = self.postconv(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out
