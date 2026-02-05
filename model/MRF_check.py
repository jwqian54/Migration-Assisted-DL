import torch
import torch.nn as nn
import torch.nn.functional as F

from model.UNet import CBAM

IMG_SIZEX = 128
IMG_SIZEY = 128

# this mutliscale class is the basic bolck whihc is defined for the multiple receptive field.
# the channel of the output of the MultiScale is 4*out_channels due to the concatenation
class MultiScale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, strides=1):
        super(MultiScale, self).__init__()
        f = int(out_channels/4)
        self.conv1 = nn.Conv2d(in_channels, f, kernel_size=1, padding=0, stride=strides)

        self.conv3 = nn.Conv2d(in_channels, f, kernel_size=kernel_size, padding=padding, stride=strides)

        self.conv5_1 = nn.Conv2d(in_channels, f, kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv5_2 = nn.Conv2d(f, f, kernel_size=kernel_size, padding=padding, stride=strides)

        self.conv7_1 = nn.Conv2d(in_channels, f, kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv7_2 = nn.Conv2d(f, f, kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv7_3 = nn.Conv2d(f, f, kernel_size=kernel_size, padding=padding, stride=strides)

        self.relu = nn.ReLU()


    def forward(self, x):
        c1 = self.relu(self.conv1(x))

        c3 = self.relu(self.conv3(x))

        c5 = self.relu(self.conv5_1(x))
        c5 = self.relu(self.conv5_2(c5))

        c7 = self.relu(self.conv7_1(x))
        c7 = self.relu(self.conv7_2(c7))
        c7 = self.relu(self.conv7_3(c7))
        # dim = 1 means the concatenation along channel dimension
        c = torch.cat([c1, c3, c5, c7], dim=1)
        return c

class DownBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, strides=1):
        super(DownBlock2, self).__init__()
        f = int(out_channels / 4)
        self.multi_scale1 = MultiScale(in_channels, out_channels, kernel_size, padding, strides)
        self.multi_scale2 = MultiScale(out_channels, out_channels, kernel_size, padding, strides)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        c = self.multi_scale1(x)
        c = self.conv1(c)
        c = self.relu(c)
        c = self.multi_scale2(c)
        c = self.conv2(c)
        c = self.relu(c)
        # c = F.relu(self.conv(c))
        p = self.pool(c)
        return c, p

class DownBlock2_w_downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, strides=1):
        super(DownBlock2_w_downsample, self).__init__()
        f = int(out_channels / 4)
        self.multi_scale1 = MultiScale(in_channels, out_channels, kernel_size, padding, strides)
        self.multi_scale2 = MultiScale(out_channels, out_channels, kernel_size, padding, strides)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        c = self.multi_scale1(x)
        c = self.conv1(c)
        c = self.relu(c)
        c = self.multi_scale2(c)
        c = self.conv2(c)
        c = self.relu(c)
        # c = F.relu(self.conv(c))
        c = self.pool(c)
        return c


class DownBlock2_wo_downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, strides=1):
        super(DownBlock2_wo_downsample, self).__init__()
        f = int(out_channels / 4)
        self.multi_scale1 = MultiScale(in_channels, out_channels, kernel_size, padding, strides)
        self.multi_scale2 = MultiScale(out_channels, out_channels, kernel_size, padding, strides)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        c = self.multi_scale1(x)
        c = self.conv1(c)
        c = self.relu(c)
        c = self.multi_scale2(c)
        c = self.conv2(c)
        c = self.relu(c)
        # c = F.relu(self.conv(c))
        return c

class UpBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, strides=1):
        super(UpBlock2, self).__init__()

        f = int(out_channels / 4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same', stride=1)
        self.multi_scale1 = MultiScale(in_channels , out_channels, kernel_size, padding, strides)
        self.multi_scale2 = MultiScale(out_channels , out_channels, kernel_size, padding, strides)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)

    def forward(self, x, skip):
        us = self.upsample(x)
        us = F.relu(self.conv_up(us))
        c = torch.cat([us, skip], dim=1)
        c = self.multi_scale1(c)
        c = F.relu(self.conv1(c))
        c = self.multi_scale2(c)
        c = F.relu(self.conv2(c))
        return c

class UpBlock2_w_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, strides=1):
        super(UpBlock2_w_CBAM, self).__init__()

        f = int(out_channels / 4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same', stride=1)
        self.multi_scale1 = MultiScale(in_channels , out_channels, kernel_size, padding, strides)
        self.multi_scale2 = MultiScale(out_channels , out_channels, kernel_size, padding, strides)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)

        self.cbam = CBAM(2*out_channels)

    def forward(self, x, skip):
        us = self.upsample(x)
        us = F.relu(self.conv_up(us))
        c = torch.cat([us, skip], dim=1)
        c = self.cbam(c)
        c = self.multi_scale1(c)
        c = F.relu(self.conv1(c))
        c = self.multi_scale2(c)
        c = F.relu(self.conv2(c))
        return c

class Bottleneck2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, strides=1):
        super(Bottleneck2, self).__init__()
        f = int(out_channels / 4)
        self.multi_scale1 = MultiScale(in_channels, out_channels, kernel_size, padding, strides)
        self.multi_scale2 = MultiScale(out_channels, out_channels, kernel_size, padding, strides)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
        self.relu = nn.ReLU()

    def forward(self, x):
        c = self.multi_scale1(x)
        c = self.relu(self.conv1(c))

        c = self.multi_scale2(c)
        c = self.relu(self.conv2(c))
        return c

class DMRFUNet1Layer(nn.Module):
    def __init__(self):
        super(DMRFUNet1Layer, self).__init__()
        f0 = 64
        f = [f0, f0 * 2, f0 * 4, f0 * 8, f0 * 16]
        self.down_block1 = DownBlock2(1, f[0])
        self.down_block2 = DownBlock2(f[0], f[1])
        self.down_block3 = DownBlock2(f[1], f[2])
        self.down_block4 = DownBlock2(f[2], f[3])
        self.bottleneck = Bottleneck2(f[3], f[4])
        self.up_block1 = UpBlock2(f[4], f[3])
        self.up_block2 = UpBlock2(f[3], f[2])
        self.up_block3 = UpBlock2(f[2], f[1])
        self.up_block4 = UpBlock2(f[1], f[0])
        self.output_layer = nn.Conv2d(f[0], 1, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1, p1 = self.down_block1(x)
        c2, p2 = self.down_block2(p1)
        c3, p3 = self.down_block3(p2)
        c4, p4 = self.down_block4(p3)
        bn = self.bottleneck(p4)
        u1 = self.up_block1(bn, c4)
        u2 = self.up_block2(u1, c3)
        u3 = self.up_block3(u2, c2)
        u4 = self.up_block4(u3, c1)
        output = self.output_layer(u4)
        output = self.relu(output)
        return output


class DMRFUNet1LayerV2(nn.Module):
    def __init__(self):
        super(DMRFUNet1LayerV2, self).__init__()
        f0 = 64
        f = [f0, f0 * 2, f0 * 4, f0 * 8, f0 * 16]
        self.down_block1 = DownBlock2(1, f[0])
        self.down_block2 = DownBlock2(f[0], f[1])
        self.down_block3 = DownBlock2(f[1], f[2])
        self.down_block4 = DownBlock2(f[2], f[3])
        self.bottleneck = Bottleneck2(f[3], f[4])
        self.up_block1 = UpBlock2_w_CBAM(f[4], f[3])
        self.up_block2 = UpBlock2_w_CBAM(f[3], f[2])
        self.up_block3 = UpBlock2_w_CBAM(f[2], f[1])
        self.up_block4 = UpBlock2_w_CBAM(f[1], f[0])
        self.output_layer = nn.Conv2d(f[0], 1, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1, p1 = self.down_block1(x)
        c2, p2 = self.down_block2(p1)
        c3, p3 = self.down_block3(p2)
        c4, p4 = self.down_block4(p3)
        bn = self.bottleneck(p4)
        u1 = self.up_block1(bn, c4)
        u2 = self.up_block2(u1, c3)
        u3 = self.up_block3(u2, c2)
        u4 = self.up_block4(u3, c1)
        output = self.output_layer(u4)
        output = self.relu(output)
        return output
# Instantiate the model
# model = DMRFUNet1Layer()
