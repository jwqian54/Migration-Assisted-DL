import torch
import torch.nn as nn
import torch.nn.functional as F


class Recurrent_blockV1(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(Recurrent_blockV1, self).__init__()
        self.t = t
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1

class DUBV2(nn.Module):
    def __init__(self, in_channels, out_channels,H):
        super(DUBV2, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,1),nn.BatchNorm2d(out_channels))

        self.pool = nn.AdaptiveMaxPool2d((H,H))

    def forward(self, x):

        x = self.conv1_1(x)
        x = self.pool(x)

        return x

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

class SkipIntepretationConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipIntepretationConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #p = self.pool(x)
        return x

class SkipIntepretationConvCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipIntepretationConvCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cbam = CBAM(out_channels)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.cbam(x)
        #p = self.pool(x)
        return x

class SkipIntepretation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipIntepretation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x)+x)
        x = F.relu(self.conv2(x)+x)
        #p = self.pool(x)
        return x

class SkipIntepretationDynamic(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer):
        super(SkipIntepretationDynamic, self).__init__()
        self.num_layer = num_layer

        self.conv = nn.ModuleList()

        for i in range(self.num_layer):
            if i ==0:
                input_channels = in_channels
            else:
                input_channels = out_channels
            self.conv.append(nn.Conv2d(input_channels,out_channels,kernel_size=3,padding=1))


        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        for i, conv in enumerate(self.conv):
            x = F.relu(conv(x)+x)

        # x = F.relu(self.conv1(x)+x)
        # x = F.relu(self.conv2(x)+x)
        #p = self.pool(x)
        return x

class SkipIntepretationDynamic2(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer):
        super(SkipIntepretationDynamic2, self).__init__()
        self.num_layer = num_layer

        self.conv = nn.ModuleList()

        self.bn = nn.ModuleList()

        for i in range(self.num_layer):
            if i ==0:
                input_channels = in_channels
            else:
                input_channels = out_channels
            self.conv.append(nn.Conv2d(input_channels,out_channels,kernel_size=3,padding=1))
            self.bn.append((nn.BatchNorm2d(out_channels)))


        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        for i, (conv,bn) in enumerate(zip(self.conv, self.bn)):
            x = F.relu(bn(conv(x))+x)

        # x = F.relu(self.conv1(x)+x)
        # x = F.relu(self.conv2(x)+x)
        #p = self.pool(x)
        return x

class SkipIntepretationDynamic3(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer):
        super(SkipIntepretationDynamic3, self).__init__()
        self.num_layer = num_layer

        self.conv1 = nn.ModuleList()

        self.bn1 = nn.ModuleList()

        self.conv2 = nn.ModuleList()

        self.bn2 = nn.ModuleList()

        for i in range(self.num_layer):
            if i ==0:
                input_channels = in_channels
            else:
                input_channels = out_channels
            self.conv1.append(nn.Conv2d(input_channels,out_channels,kernel_size=3,padding=1))
            self.bn1.append(nn.BatchNorm2d(out_channels))

            self.conv2.append(nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1))
            self.bn2.append(nn.BatchNorm2d(out_channels))


        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        for i, (conv1,bn1,conv2,bn2) in enumerate(zip(self.conv1, self.bn1, self.conv2, self.bn2)):
            x1 = F.relu(bn1(conv1(x)))

            x1 = bn2(conv2(x1))

        # x = F.relu(self.conv1(x)+x)
        # x = F.relu(self.conv2(x)+x)
        #p = self.pool(x)
        return F.relu(x1+x)

class SkipIntepretationDynamic4(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer):
        super(SkipIntepretationDynamic4, self).__init__()
        self.num_layer = num_layer

        self.conv1 = nn.ModuleList()

        # self.bn1 = nn.ModuleList()

        self.conv2 = nn.ModuleList()

        # self.bn2 = nn.ModuleList()

        for i in range(self.num_layer):
            if i ==0:
                input_channels = in_channels
            else:
                input_channels = out_channels
            self.conv1.append(nn.Conv2d(input_channels,out_channels,kernel_size=3,padding=1))
            # self.bn1.append(nn.BatchNorm2d(out_channels))

            self.conv2.append(nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1))
            # self.bn2.append(nn.BatchNorm2d(out_channels))


        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        for i, (conv1,conv2) in enumerate(zip(self.conv1, self.conv2)):
            x1 = F.relu((conv1(x)))

            x1 = conv2(x1)

        # x = F.relu(self.conv1(x)+x)
        # x = F.relu(self.conv2(x)+x)
        #p = self.pool(x)
        return F.relu(x1+x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')
        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip and x
        self.conv2 = nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x



class UpBlock_wo_skip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock_wo_skip, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')
        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip and x
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        #x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class UpBlock_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock_CBAM, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')

        self.cbam = CBAM(2*out_channels)
        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip and x
        self.conv2 = nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        x = self.cbam(x)
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


class BottleneckV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.sab = SAB(out_channels,out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.sab(x))
        return x

class BottleneckV3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckV3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.cbam(x))
        return x
# UNet model
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
        # final_conv also need modify, out_channels is just the 1
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

class UNetV19(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV19, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.skip1 = SkipIntepretationConvCBAM(filters[0],filters[0])
        self.skip2 = SkipIntepretationConvCBAM(filters[1],filters[1])
        self.skip3 = SkipIntepretationConvCBAM(filters[2],filters[2])
        self.skip4 = SkipIntepretationConvCBAM(filters[3],filters[3])

        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV18(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV18, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.skip1 = SkipIntepretationConv(filters[0],filters[0])
        self.skip2 = SkipIntepretationConv(filters[1],filters[1])
        self.skip3 = SkipIntepretationConv(filters[2],filters[2])
        self.skip4 = SkipIntepretationConv(filters[3],filters[3])

        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV17(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV17, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.skip1 = SkipIntepretation(filters[0],filters[0])
        self.skip2 = SkipIntepretation(filters[1],filters[1])
        self.skip3 = SkipIntepretation(filters[2],filters[2])
        self.skip4 = SkipIntepretation(filters[3],filters[3])

        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV15(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV15, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.skip1 = SkipIntepretationDynamic3(filters[0],filters[0],1)
        self.skip2 = SkipIntepretationDynamic3(filters[1],filters[1],1)
        self.skip3 = SkipIntepretationDynamic3(filters[2],filters[2],1)
        self.skip4 = SkipIntepretationDynamic3(filters[3],filters[3],1)

        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV16(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV16, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.skip1 = SkipIntepretationDynamic4(filters[0],filters[0],1)
        self.skip2 = SkipIntepretationDynamic4(filters[1],filters[1],1)
        self.skip3 = SkipIntepretationDynamic4(filters[2],filters[2],1)
        self.skip4 = SkipIntepretationDynamic4(filters[3],filters[3],1)

        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV13(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV13, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.skip1 = SkipIntepretationDynamic3(filters[0],filters[0],4)
        self.skip2 = SkipIntepretationDynamic3(filters[1],filters[1],3)
        self.skip3 = SkipIntepretationDynamic3(filters[2],filters[2],2)
        self.skip4 = SkipIntepretationDynamic3(filters[3],filters[3],1)

        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV14(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV14, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.skip1 = SkipIntepretationDynamic2(filters[0],filters[0],4)
        self.skip2 = SkipIntepretationDynamic2(filters[1],filters[1],3)
        self.skip3 = SkipIntepretationDynamic2(filters[2],filters[2],2)
        self.skip4 = SkipIntepretationDynamic2(filters[3],filters[3],1)

        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV11(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV11, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.skip1 = SkipIntepretation(filters[0],filters[0])
        self.skip2 = SkipIntepretation(filters[1],filters[1])
        self.skip3 = SkipIntepretation(filters[2],filters[2])
        self.skip4 = SkipIntepretation(filters[3],filters[3])

        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV12(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV12, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.skip1 = SkipIntepretationDynamic(filters[0],filters[0],4)
        self.skip2 = SkipIntepretationDynamic(filters[1],filters[1],3)
        self.skip3 = SkipIntepretationDynamic(filters[2],filters[2],2)
        self.skip4 = SkipIntepretationDynamic(filters[3],filters[3],1)

        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV10(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV10, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.up1 = UpBlock_wo_skip(filters[4], filters[3])
        self.up2 = UpBlock_wo_skip(filters[3], filters[2])
        self.up3 = UpBlock_wo_skip(filters[2], filters[1])
        self.up4 = UpBlock_wo_skip(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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



class UNetV9(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV9, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3], filters[4])

        self.up1 = UpBlock_CBAM(filters[4], filters[3])
        self.up2 = UpBlock_CBAM(filters[3], filters[2])
        self.up3 = UpBlock_CBAM(filters[2], filters[1])
        self.up4 = UpBlock_CBAM(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV8, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.dub1 = DUBV2(filters[0], filters[0],8)
        self.dub2 = DUBV2(filters[1], filters[1], 8)
        self.dub3 = DUBV2(filters[2], filters[2], 8)
        self.dub4 = DUBV2(filters[3], filters[3], 8)


        self.cbam = CBAM(filters[0]+filters[1]+filters[2]+filters[3])
        self.bottleneck = Bottleneck(filters[0]+filters[1]+filters[2]+filters[3], filters[4])

        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):

        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)



        p = torch.cat([self.dub1(p1),self.dub2(p2),self.dub3(p3),self.dub4(p4)],dim = 1)
        p = self.cbam(p)

        bn = self.bottleneck(p)

        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)

        outputs = F.relu(self.final_conv(u4))
        return outputs


class UNetBCE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBCE, self).__init__()
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
        # final_conv also need modify, out_channels is just the 1
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

        outputs = F.sigmoid(self.final_conv(u4))
        return outputs

class UNetV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV2, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = BottleneckV2(filters[3], filters[4])

        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV3, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = BottleneckV3(filters[3], filters[4])

        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV4, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = BottleneckV3(filters[3], filters[4])

        self.cbam1 = CBAM(filters[0])
        self.cbam2 = CBAM(filters[1])
        self.cbam3 = CBAM(filters[2])



        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):

        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)

        bn = self.bottleneck(p4)

        c3 = F.relu(self.cbam3(c3))
        c2 = F.relu(self.cbam2(c2))
        c1 = F.relu(self.cbam1(c1))

        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)

        outputs = F.relu(self.final_conv(u4))
        return outputs

class UNetV5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetV5, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock_wo_SAB(filters[0], filters[1])
        self.down3 = DownBlock_wo_SAB(filters[1], filters[2])
        self.down4 = DownBlock_wo_SAB(filters[2], filters[3])

        self.bottleneck = BottleneckV3(filters[3], filters[4])

        self.up1 = UpBlock_wo_SAB(filters[4], filters[3])
        self.up2 = UpBlock_wo_SAB(filters[3], filters[2])
        self.up3 = UpBlock_wo_SAB(filters[2], filters[1])
        self.up4 = UpBlock_wo_SAB(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV6(nn.Module):
    def __init__(self, in_channels, out_channels,atrous_rates = [6,12,18]):
        super(UNetV6, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck_w_ASPP(filters[3], filters[4],atrous_rates=atrous_rates)

        self.up1 = UpBlock_wo_SAB(filters[4], filters[3])
        self.up2 = UpBlock_wo_SAB(filters[3], filters[2])
        self.up3 = UpBlock_wo_SAB(filters[2], filters[1])
        self.up4 = UpBlock_wo_SAB(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV6_2(nn.Module):
    def __init__(self, in_channels, out_channels,atrous_rates = [6,12,18]):
        super(UNetV6_2, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck_w_ASPP(filters[3], filters[4],atrous_rates=atrous_rates)

        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNetV7(nn.Module):
    def __init__(self, in_channels, out_channels,atrous_rates = [6,12,18]):
        super(UNetV7, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = DownBlock_w_ASPP(in_channels, filters[0],atrous_rates=atrous_rates)
        self.down2 = DownBlock_w_ASPP(filters[0], filters[1],atrous_rates=atrous_rates)
        self.down3 = DownBlock_w_ASPP(filters[1], filters[2],atrous_rates=atrous_rates)
        self.down4 = DownBlock_w_ASPP(filters[2], filters[3],atrous_rates=atrous_rates)

        self.bottleneck = Bottleneck_w_ASPP(filters[3], filters[4],atrous_rates=atrous_rates)

        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UpBlock_RefStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock_RefStack, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')
        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip1,skip2 and x
        self.conv2 = nn.Conv2d(3*out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip1,skip2):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip1,skip2], dim=1)  # Concatenate along the channel axis
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class UNet_RefStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_RefStack, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down_ori1 = DownBlock(in_channels, filters[0])
        self.down_ori2 = DownBlock(filters[0], filters[1])
        self.down_ori3 = DownBlock(filters[1], filters[2])
        self.down_ori4 = DownBlock(filters[2], filters[3])

        self.down_mig1 = DownBlock(in_channels, filters[0])
        self.down_mig2 = DownBlock(filters[0], filters[1])
        self.down_mig3 = DownBlock(filters[1], filters[2])
        self.down_mig4 = DownBlock(filters[2], filters[3])

        self.bottleneck = Bottleneck(filters[3]*2, filters[4])

        self.up1 = UpBlock_RefStack(filters[4], filters[3])
        self.up2 = UpBlock_RefStack(filters[3], filters[2])
        self.up3 = UpBlock_RefStack(filters[2], filters[1])
        self.up4 = UpBlock_RefStack(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x_ori,x_mig):

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

        u1 = self.up1(bn, skip_mig4,skip_ori4)
        u2 = self.up2(u1, skip_mig3,skip_ori3)
        u3 = self.up3(u2, skip_mig2,skip_ori2)
        u4 = self.up4(u3, skip_mig1,skip_ori1)

        outputs = F.relu(self.final_conv(u4))
        return outputs


class SpatialAttention_wo_skip(nn.Module):
    def __init__(self, in_channels, out_channels,channel_reduction_rate = 8):
        super(SpatialAttention_wo_skip, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels, out_channels/channel_reduction_rate, kernel_size=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels, out_channels/channel_reduction_rate, kernel_size=1, padding='same')

    def forward(self, x):
        v = self.conv1(x)
        v = v.view(v.size(0),v.size(1),-1)
        q = self.conv2(x)
        q = q.view(q.size(0),q.size(1),-1)
        q = q.transpose(1,2)
        k = self.conv3(x)
        k = k.view(k.size(0),k.size(1),-1)
        weight = F.softmax(torch.matmul(q,k))
        weighted_v = torch.matmul(v,weight)
        weighted_v = weighted_v.reshape(x.shape)
        return weighted_v

class SpatialAttention_w_skip(nn.Module):
    def __init__(self, in_channels, out_channels,channel_reduction_rate = 8):
        super(SpatialAttention_w_skip, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels, int(out_channels/channel_reduction_rate), kernel_size=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels, int(out_channels/channel_reduction_rate), kernel_size=1, padding='same')

    def forward(self, x):
        v = self.conv1(x)
        v = v.view(v.size(0),v.size(1),-1)
        q = self.conv2(x)
        q = q.view(q.size(0),q.size(1),-1)
        q = q.transpose(1,2)
        k = self.conv3(x)
        k = k.view(k.size(0),k.size(1),-1)
        weight_v = torch.bmm(q,k)
        size = weight_v.shape
        weight_v = F.softmax(weight_v.view(x.size(0),-1),dim=1).view(size)

        weight_v = torch.bmm(v,weight_v)
        weight_v = weight_v.reshape(x.shape)
        return weight_v+x

class SpatialAttention_w_skipV2(nn.Module):
    def __init__(self, in_channels, out_channels,channel_reduction_rate = 8):
        super(SpatialAttention_w_skipV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 2, padding=1)
        self.conv2 = nn.Conv2d(in_channels, int(out_channels/channel_reduction_rate), kernel_size=3,stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels, int(out_channels/channel_reduction_rate), kernel_size=3,stride=2, padding=1)

    def forward(self, x):
        v = self.conv1(x)
        v = v.view(v.size(0),v.size(1),-1)
        q = self.conv2(x)
        q = q.view(q.size(0),q.size(1),-1)
        q = q.transpose(1,2)
        k = self.conv3(x)
        k = k.view(k.size(0),k.size(1),-1)
        weight_v = torch.bmm(q,k)
        size = weight_v.shape
        weight_v = F.softmax(weight_v.view(x.size(0),-1),dim=1).view(size)
        weight_v = torch.bmm(v,weight_v)


        weight_v = weight_v.reshape(x.size(0),x.size(1),int(x.size(2)/2),int(x.size(3)/2))
        weight_v = F.interpolate(weight_v,scale_factor=2,mode='bilinear')
        return weight_v+x


class GlobalChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')

    def forward(self, x):
        v = self.conv1(x)
        v = v.view(v.size(0),v.size(1),-1).transpose(1,2) # after transpose, N*HW*C
        q = self.conv2(x)
        q = q.view(q.size(0),q.size(1),-1)
        k = self.conv3(x)
        k = k.view(k.size(0),k.size(1),-1)
        k = k.transpose(1, 2)
        weight = torch.bmm(q,k)
        weight = F.softmax(weight.view(weight.size(0),-1),dim=1).view(weight.shape)
        weighted_v = torch.bmm(v,weight).transpose(1,2) #after transpose N*C*HW
        weighted_v = weighted_v.reshape(x.shape)
        return weighted_v

class GlobalChannelAttention2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalChannelAttention2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')

    def forward(self, x):
        v = self.conv1(x)
        v = v.view(v.size(0),v.size(1),-1).transpose(1,2) # after transpose, N*HW*C
        q = self.conv2(x)
        q = q.view(q.size(0),q.size(1),-1)
        k = self.conv3(x)
        k = k.view(k.size(0),k.size(1),-1)
        k = k.transpose(1, 2)
        weight = torch.bmm(q,k)
        weight = F.softmax(weight.view(weight.size(0),-1),dim=1).view(weight.shape)
        weighted_v = torch.bmm(v,weight).transpose(1,2) #after transpose N*C*HW
        weighted_v = weighted_v.reshape(x.shape)
        return weighted_v

class LocalChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same'), nn.AdaptiveAvgPool2d(1) )

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same'), nn.AdaptiveMaxPool2d(1) )

        self.conv4 = nn.Conv2d(out_channels*2, out_channels,kernel_size=1,padding='same')


    def forward(self, x):
        v = self.conv1(x)
        q = self.conv2(x)
        k = self.conv3(x)
        weight = torch.cat((q,k),dim=1)
        weight = self.conv4(weight)

        weight = F.softmax(weight, dim =1)
        return v*weight

class SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAB, self).__init__()

        self.GCA = GlobalChannelAttention(in_channels, out_channels)
        self.LCA = LocalChannelAttention(in_channels,out_channels)
        self.SA = SpatialAttention_w_skip(in_channels,out_channels)

    def forward(self, x):

        out = self.GCA(x)+self.LCA(x)+self.SA(x)

        return out

class SABV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SABV2, self).__init__()

        self.GCA = GlobalChannelAttention(in_channels, out_channels)
        self.LCA = LocalChannelAttention(in_channels,out_channels)
        self.SA = SpatialAttention_w_skipV2(in_channels,out_channels)

    def forward(self, x):

        out = self.GCA(x)+self.LCA(x)+self.SA(x)

        return out

class ResBlcok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1_1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding='same')


    def forward(self, x):


        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        x = self.conv1_1(x)


        out = F.relu(out+x)

        return out

class DownBlock_w_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock_w_SAB, self).__init__()
        self.res1 = ResBlcok(in_channels,out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.SAB = SAB(out_channels,out_channels)

    def forward(self, x):

        x = self.res1(x)
        x = self.SAB(x)

        p = self.pool(x)
        return x, p

class UpBlock_w_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock_w_SAB, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')


        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip and x
        self.res1 = ResBlcok(2*out_channels,out_channels)
        self.sab = SAB(out_channels,out_channels)


    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        x = self.res1(x)
        x = self.sab(x)
        return x

class Bottleneck_w_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck_w_SAB, self).__init__()

        self.res1 = ResBlcok(in_channels,out_channels)
        self.sab = SAB(out_channels,out_channels)

    def forward(self, x):
        x = self.res1(x)
        x = self.sab(x)
        return x

class DownBlock_w_SABV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock_w_SABV2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.SAB = SABV2(out_channels,out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.SAB(x)

        p = self.pool(x)
        return x, p


class UpBlock_w_SABV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock_w_SABV2, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')
        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip and x
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.sab = SABV2(out_channels,out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.sab(x)
        return x

        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip and x


class Bottleneck_w_SABV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck_w_SABV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.sab = SABV2(out_channels,out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.sab(x)
        return x


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size =3, dilation= 4, ):
        super(ASPPConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPooling, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(in_channels,out_channels,1))

        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        size = x.shape[-2:]

        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6,12,18]):
        super(ASPP, self).__init__()
        rate1, rate2, rate3 = tuple(atrous_rates)

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels,in_channels,kernel_size=1),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())

        self.asppconv1 = ASPPConv(in_channels, in_channels,kernel_size=3,dilation= rate1)
        self.asppconv2 = ASPPConv(in_channels,in_channels,kernel_size=3,dilation=rate2)
        self.asppconv3 = ASPPConv(in_channels,in_channels,kernel_size=3,dilation=rate3)

        self.asppooling = ASPPooling(in_channels,in_channels)

        self.conv_post = nn.Sequential(nn.Conv2d(5*in_channels,out_channels,1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))

    def forward(self, x):

        out = self.conv1_1(x)
        out2 = self.asppconv1(x)
        out3 = self.asppconv2(x)
        out4 = self.asppconv3(x)
        out5 = self.asppooling(x)

        out = torch.cat([out,out2,out3,out4,out5],dim=1)

        out = self.conv_post(out)

        return out


class Bottleneck_w_ASPP(nn.Module):
    def __init__(self, in_channels, out_channels,atrous_rates=[6,12,18]):
        super(Bottleneck_w_ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.aspp = ASPP(out_channels,out_channels,atrous_rates=atrous_rates)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.aspp(x)
        return x

class DownBlock_w_ASPP(nn.Module):
    def __init__(self, in_channels, out_channels,atrous_rates = [6,12,18]):
        super(DownBlock_w_ASPP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.aspp = ASPP(out_channels,out_channels,atrous_rates=atrous_rates)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.aspp(x)

        p = self.pool(x)
        return x, p

class UpBlock_w_ASPP(nn.Module):
    def __init__(self, in_channels, out_channels,atrous_rates = [6,12,18]):
        super(UpBlock_w_ASPP, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')
        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip and x
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.aspp = ASPP(out_channels,out_channels,atrous_rates=atrous_rates)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.aspp(x)
        return x

        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip and x

class DownBlock_wo_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock_wo_SAB, self).__init__()
        self.res1 = ResBlcok(in_channels,out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.SAB = SAB(out_channels,out_channels)

    def forward(self, x):

        x = self.res1(x)
        # x = self.SAB(x)

        p = self.pool(x)
        return x, p

class UpBlock_wo_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock_wo_SAB, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')


        # the input of the first variable of the conv2. should be 2*out_channels, cuz the input is the concatenation of both skip and x
        self.res1 = ResBlcok(2*out_channels,out_channels)
        self.sab = SAB(out_channels,out_channels)


    def forward(self, x, skip):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        x = self.res1(x)
        # x = self.sab(x)
        return x

class Bottleneck_wo_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck_wo_SAB, self).__init__()

        self.res1 = ResBlcok(in_channels,out_channels)
        self.sab = SAB(out_channels,out_channels)

    def forward(self, x):
        x = self.res1(x)
        # x = self.sab(x)
        return x

class UNet_wo_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_wo_SAB, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        # filters = [32,64, 128, 256, 512]
        self.down1 = DownBlock_wo_SAB(in_channels, filters[0])
        self.down2 = DownBlock_wo_SAB(filters[0], filters[1])
        self.down3 = DownBlock_wo_SAB(filters[1], filters[2])
        self.down4 = DownBlock_wo_SAB(filters[2], filters[3])

        self.bottleneck = Bottleneck_wo_SAB(filters[3], filters[4])

        self.up1 = UpBlock_wo_SAB(filters[4], filters[3])
        self.up2 = UpBlock_wo_SAB(filters[3], filters[2])
        self.up3 = UpBlock_wo_SAB(filters[2], filters[1])
        self.up4 = UpBlock_wo_SAB(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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
# UNet model
class UNet_w_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_w_SAB, self).__init__()
        # filters = [64, 128, 256, 512, 1024]
        filters = [32,64, 128, 256, 512]
        self.down1 = DownBlock_w_SAB(in_channels, filters[0])
        self.down2 = DownBlock_w_SAB(filters[0], filters[1])
        self.down3 = DownBlock_w_SAB(filters[1], filters[2])
        self.down4 = DownBlock_w_SAB(filters[2], filters[3])

        self.bottleneck = Bottleneck_w_SAB(filters[3], filters[4])

        self.up1 = UpBlock_w_SAB(filters[4], filters[3])
        self.up2 = UpBlock_w_SAB(filters[3], filters[2])
        self.up3 = UpBlock_w_SAB(filters[2], filters[1])
        self.up4 = UpBlock_w_SAB(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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

class UNet_w_SABV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_w_SABV2, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        # filters = [32,64, 128, 256, 512]
        self.down1 = DownBlock_w_SABV2(in_channels, filters[0])
        self.down2 = DownBlock_w_SABV2(filters[0], filters[1])
        self.down3 = DownBlock_w_SABV2(filters[1], filters[2])
        self.down4 = DownBlock_w_SABV2(filters[2], filters[3])

        self.bottleneck = Bottleneck_w_SABV2(filters[3], filters[4])

        self.up1 = UpBlock_w_SABV2(filters[4], filters[3])
        self.up2 = UpBlock_w_SABV2(filters[3], filters[2])
        self.up3 = UpBlock_w_SABV2(filters[2], filters[1])
        self.up4 = UpBlock_w_SABV2(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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


class UNet_w_BottleSAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_w_BottleSAB, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        # filters = [32,64, 128, 256, 512]
        self.down1 = DownBlock_wo_SAB(in_channels, filters[0])
        self.down2 = DownBlock_wo_SAB(filters[0], filters[1])
        self.down3 = DownBlock_wo_SAB(filters[1], filters[2])
        self.down4 = DownBlock_wo_SAB(filters[2], filters[3])

        self.bottleneck = Bottleneck_w_SAB(filters[3], filters[4])

        self.up1 = UpBlock_wo_SAB(filters[4], filters[3])
        self.up2 = UpBlock_wo_SAB(filters[3], filters[2])
        self.up3 = UpBlock_wo_SAB(filters[2], filters[1])
        self.up4 = UpBlock_wo_SAB(filters[1], filters[0])
        # final_conv also need modify, out_channels is just the 1
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


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
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
        # map尺寸不变，缩减通道
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