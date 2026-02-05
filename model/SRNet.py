"""SRNet: single model file, no cross-import from other model modules."""
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


class SkipIntepretation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipIntepretation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x) + x)
        x = F.relu(self.conv2(x) + x)
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


class UpBlock_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock_CBAM, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')
        self.cbam = CBAM(2 * out_channels)
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip], dim=1)
        x = self.cbam(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class SRNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SRNet, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])
        self.bottleneck = Bottleneck(filters[3], filters[4])
        self.skip1 = SkipIntepretation(filters[0], filters[0])
        self.skip2 = SkipIntepretation(filters[1], filters[1])
        self.skip3 = SkipIntepretation(filters[2], filters[2])
        self.skip4 = SkipIntepretation(filters[3], filters[3])
        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)
        bn = self.bottleneck(p4)
        c4 = self.skip4(c4)
        c3 = self.skip3(c3)
        c2 = self.skip2(c2)
        c1 = self.skip1(c1)
        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        outputs = F.relu(self.final_conv(u4))
        return outputs
