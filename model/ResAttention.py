import torch
import torch.nn as nn
import torch.nn.functional as F
from model.UNet import CBAM
from model.UNet import SAB
from model.MRF_check import DownBlock2_wo_downsample as MRFBlock_wo_downsample
from model.MRF_check import DownBlock2_w_downsample as MRFBlock_w_downsample

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

class ResBlcok_wo_downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_wo_downsample, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        out = F.relu(out+x)
        return out

class ResBlcok_wo_downsample_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_wo_downsample_CBAM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.cbam = CBAM(out_channels)


    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)

        out = F.relu(out+x)
        return out

class ResBlcok_wo_downsample_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_wo_downsample_SAB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.sab = SAB(out_channels,out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding='same')


    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.sab(out)
        out = self.conv3(out)

        out = F.relu(out+x)
        return out

class ResBlcok_wo_downsample_MRF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_wo_downsample_MRF, self).__init__()

        self.mrf = MRFBlock_wo_downsample(in_channels,out_channels)



    def forward(self, x):

        out = self.mrf(x)

        out = F.relu(out+x)
        return out

class ResBlcok_w_downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_w_downsample, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 2),
                                     nn.BatchNorm2d(out_channels))



    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        x = self.conv1_1(x)

        out = F.relu(out + x)
        return out

class ResBlcok_w_downsample_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_w_downsample_CBAM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,2),
                                     nn.BatchNorm2d(out_channels))

        self.cbam = CBAM(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)

        x = self.conv1_1(x)

        out = F.relu(out + x)
        return out

class ResBlcok_w_downsample_SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_w_downsample_SAB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,2),
                                     nn.BatchNorm2d(out_channels))

        self.sab= SAB(out_channels,out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels,3,1,1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.sab(out)
        out = self.conv3(out)

        x = self.conv1_1(x)

        out = F.relu(out + x)
        return out

class ResBlcok_w_downsample_MRF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlcok_w_downsample_MRF, self).__init__()

        self.mrf = MRFBlock_wo_downsample(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2,2)

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,2),
                                     nn.BatchNorm2d(out_channels))

    def forward(self, x):

        out = self.mrf(x)
        out = self.pool(out)
        x = self.conv1_1(x)

        out = F.relu(out + x)
        return out

class DUBV1(nn.Module):
    def __init__(self, in_channels, out_channels,H):
        super(DUBV1, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,1),nn.BatchNorm2d(out_channels))

        self.pool = nn.AdaptiveAvgPool2d((H,H))

        self.smooth = nn.Sequential(nn.Conv2d(out_channels,out_channels,3,1,1),nn.BatchNorm2d(out_channels),nn.ReLU())


    def forward(self, x):

        x = self.conv1_1(x)
        x = self.pool(x)
        x = self.smooth(x)

        return x

class DUBV2(nn.Module):
    def __init__(self, in_channels, out_channels,H):
        super(DUBV2, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,1),nn.BatchNorm2d(out_channels))

        self.pool = nn.AdaptiveMaxPool2d((H,H))

    def forward(self, x):

        x = self.conv1_1(x)
        x = self.pool(x)

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

class ASPP_w_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6,12,18]):
        super(ASPP_w_CBAM, self).__init__()
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
        self.CBAM = CBAM(5*in_channels)

    def forward(self, x):

        out = self.conv1_1(x)
        out2 = self.asppconv1(x)
        out3 = self.asppconv2(x)
        out4 = self.asppconv3(x)
        out5 = self.asppooling(x)

        out = torch.cat([out,out2,out3,out4,out5],dim=1)
        out = self.CBAM(out)

        out = self.conv_post(out)

        return out

class Para2outV1(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV1, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1 = nn.Sequential(ResBlcok(out_channels[0],out_channels[0]),
                                       ResBlcok(out_channels[0],out_channels[0]),
                                       nn.MaxPool2d(2))
        self.resblock2 = nn.Sequential(ResBlcok(out_channels[0],out_channels[1]),
                                       ResBlcok(out_channels[1],out_channels[1]),
                                       nn.MaxPool2d(2))
        self.resblock3 = nn.Sequential(ResBlcok(out_channels[1],out_channels[2]),
                                       ResBlcok(out_channels[2],out_channels[2]),
                                       nn.MaxPool2d(2))
        self.resblock4 = nn.Sequential(ResBlcok(out_channels[2],out_channels[3]),
                                       ResBlcok(out_channels[3],out_channels[3]),
                                       )

        self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))


        self.linear = nn.Sequential(nn.Linear(in_features=512,out_features=2), nn.ReLU())

    def forward(self, x):
        x = self.preconv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.aspp(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)

        return x


class Para2outV2(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV2, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1 = nn.Sequential(ResBlcok(out_channels[0],out_channels[0]),
                                       ResBlcok(out_channels[0],out_channels[0]),
                                       nn.MaxPool2d(2))
        self.resblock2 = nn.Sequential(ResBlcok(out_channels[0],out_channels[1]),
                                       ResBlcok(out_channels[1],out_channels[1]),
                                       nn.MaxPool2d(2))
        self.resblock3 = nn.Sequential(ResBlcok(out_channels[1],out_channels[2]),
                                       ResBlcok(out_channels[2],out_channels[2]),
                                       nn.MaxPool2d(2))
        self.resblock4 = nn.Sequential(ResBlcok(out_channels[2],out_channels[3]),
                                       ResBlcok(out_channels[3],out_channels[3]),
                                       nn.AdaptiveAvgPool2d((1,1)))

        self.linear = nn.Sequential(nn.Linear(in_features=512,out_features=2), nn.ReLU())

    def forward(self, x):
        x = self.preconv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = x.view(x.shape[0],-1)
        x = self.linear(x)

        return x

class Para2outV3(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV3, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]),
                                       nn.AdaptiveAvgPool2d((1,1)))

        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))


        self.linear = nn.Sequential(nn.Linear(in_features=512,out_features=2), nn.ReLU())

    def forward(self, x):
        x = self.preconv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        # x = self.aspp(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)

        return x

class Para2outV4(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV4, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample_CBAM(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample_CBAM(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample_CBAM(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample_CBAM(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample_CBAM(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample_CBAM(out_channels[2],out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample_CBAM(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample_CBAM(out_channels[3],out_channels[3]),
                                       nn.AdaptiveAvgPool2d((1,1)))

        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))


        self.linear = nn.Sequential(nn.Linear(in_features=512,out_features=2), nn.ReLU())

    def forward(self, x):
        x = self.preconv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        # x = self.aspp(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)

        return x

class Para2outV5(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV5, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample_SAB(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample_SAB(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample_SAB(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample_SAB(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample_SAB(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample_SAB(out_channels[2],out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample_SAB(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample_SAB(out_channels[3],out_channels[3]),
                                       nn.AdaptiveAvgPool2d((1,1)))

        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))


        self.linear = nn.Sequential(nn.Linear(in_features=512,out_features=2), nn.ReLU())

    def forward(self, x):
        x = self.preconv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        # x = self.aspp(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)

        return x

class Para2outV6(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV6, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1 = nn.Sequential(MRFBlock_wo_downsample(out_channels[0], out_channels[0]),
                                       MRFBlock_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(MRFBlock_w_downsample(out_channels[0], out_channels[1]),
                                       MRFBlock_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(MRFBlock_w_downsample(out_channels[1],out_channels[2]),
                                       MRFBlock_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock4 = nn.Sequential(MRFBlock_w_downsample(out_channels[2],out_channels[3]),
                                       MRFBlock_wo_downsample(out_channels[3],out_channels[3]),
                                       )

        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.linear = nn.Sequential(nn.Linear(in_features=512,out_features=2), nn.ReLU())

    def forward(self, x):
        x = self.preconv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.pool(x)
        # x = self.aspp(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)

        return x

class Para2outV7(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV7, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample_MRF(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample_MRF(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample_MRF(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample_MRF(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample_MRF(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample_MRF(out_channels[2],out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample_MRF(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample_MRF(out_channels[3],out_channels[3]),
                                       )

        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.linear = nn.Sequential(nn.Linear(in_features=512,out_features=2), nn.ReLU())

    def forward(self, x):
        x = self.preconv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.pool(x)
        # x = self.aspp(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)

        return x

class Para2outV8(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV8, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]))

        self.dub1 = DUBV1(out_channels[0],out_channels[0],4)
        self.dub2 = DUBV1(out_channels[1],out_channels[0],4)
        self.dub3 = DUBV1(out_channels[2],out_channels[0],4)
        self.dub4 = DUBV1(out_channels[3],out_channels[0],4)
        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))

        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[0]*4,out_features=2), nn.ReLU())

        self.postconv = nn.Sequential(nn.Conv2d(out_channels[0]*4,out_channels[0]*4,3,1,1),nn.BatchNorm2d(out_channels[0]*4),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

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
        # x = self.aspp(x)

        out = torch.cat([x1,x2,x3,x4],dim =1)
        out = self.postconv(out)
        out = out.view(out.shape[0],-1)
        out = self.linear(out)

        return out

class Para2outV9(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV9, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]))

        self.dub1 = DUBV2(out_channels[0],out_channels[0],4)
        self.dub2 = DUBV2(out_channels[1],out_channels[1],4)
        self.dub3 = DUBV2(out_channels[2],out_channels[2],4)
        self.dub4 = DUBV2(out_channels[3],out_channels[3],4)
        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))

        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3],out_features=2), nn.ReLU())

        self.postconv = nn.Sequential(nn.Conv2d(sum(out_channels),out_channels[3],3,1,1),nn.BatchNorm2d(out_channels[3]),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

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
        # x = self.aspp(x)

        out = torch.cat([x1,x2,x3,x4],dim =1)
        out = self.postconv(out)
        out = out.view(out.shape[0],-1)
        out = self.linear(out)

        return out


class Para2outV10(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64, 128, 256, 512], atrous_rates=[6, 12, 18]):
        super(Para2outV10, self).__init__()

        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels[0], 7, 2, padding=3),
                                      nn.BatchNorm2d(out_channels[0]), nn.ReLU(), nn.MaxPool2d(3, 2, 1))



        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                         ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                         ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1], out_channels[2]),
                                         ResBlcok_wo_downsample(out_channels[2], out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2], out_channels[3]),
                                         ResBlcok_wo_downsample(out_channels[3], out_channels[3]))



        self.dub1 = DUBV2(out_channels[0], out_channels[0], 4)
        self.dub2 = DUBV2(out_channels[1], out_channels[1], 4)
        self.dub3 = DUBV2(out_channels[2], out_channels[2], 4)
        self.dub4 = DUBV2(out_channels[3], out_channels[3], 4)


        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.cbam = CBAM(sum(out_channels))

        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3], out_features=2), nn.ReLU())

        self.postconv = nn.Sequential(nn.Conv2d(sum(out_channels), out_channels[3], 1, 1),
                                       nn.BatchNorm2d(out_channels[3]), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))



    def forward(self, x):

        x = self.preconv1(x)
        x1 = self.resblock1(
            x)
        x2 = self.resblock2(x1)
        x3 = self.resblock3(x2)
        x4 = self.resblock4(x3)


        x1 = self.dub1(x1)
        x2 = self.dub2(x2)
        x3 = self.dub3(x3)
        x4 = self.dub4(x4)
        # x = self.aspp(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.cbam(out)
        out = self.postconv(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

class Para2outV10_wo_fusion(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64, 128, 256, 512], atrous_rates=[6, 12, 18]):
        super(Para2outV10_wo_fusion, self).__init__()

        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels[0], 7, 2, padding=3),
                                      nn.BatchNorm2d(out_channels[0]), nn.ReLU(), nn.MaxPool2d(3, 2, 1))



        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                         ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                         ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1], out_channels[2]),
                                         ResBlcok_wo_downsample(out_channels[2], out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2], out_channels[3]),
                                         ResBlcok_wo_downsample(out_channels[3], out_channels[3]))



        # self.dub1 = DUBV2(out_channels[0], out_channels[0], 4)
        # self.dub2 = DUBV2(out_channels[1], out_channels[1], 4)
        # self.dub3 = DUBV2(out_channels[2], out_channels[2], 4)
        # self.dub4 = DUBV2(out_channels[3], out_channels[3], 4)


        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.cbam = CBAM(out_channels[3])

        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3], out_features=2), nn.ReLU())

        self.postconv = nn.Sequential(nn.Conv2d(out_channels[3], out_channels[3], 1, 1),
                                       nn.BatchNorm2d(out_channels[3]), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))



    def forward(self, x):

        x = self.preconv1(x)
        x1 = self.resblock1(
            x)
        x2 = self.resblock2(x1)
        x3 = self.resblock3(x2)
        x4 = self.resblock4(x3)



        # x1 = self.dub1(x1)
        # x2 = self.dub2(x2)
        # x3 = self.dub3(x3)
        # x4 = self.dub4(x4)


        # out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.cbam(x4)
        out = self.postconv(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

class Para2outV10_wo_fusionV2(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64, 128, 256, 512], atrous_rates=[6, 12, 18]):
        super(Para2outV10_wo_fusionV2, self).__init__()

        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels[0], 7, 2, padding=3),
                                      nn.BatchNorm2d(out_channels[0]), nn.ReLU(), nn.MaxPool2d(3, 2, 1))



        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                         ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                         ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1], out_channels[2]),
                                         ResBlcok_wo_downsample(out_channels[2], out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2], out_channels[3]),
                                         ResBlcok_wo_downsample(out_channels[3], out_channels[3]))


        self.cbam = CBAM(out_channels[3])

        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3], out_features=2), nn.ReLU())

        self.postconv = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))



    def forward(self, x):

        x = self.preconv1(x)
        x1 = self.resblock1(
            x)
        x2 = self.resblock2(x1)
        x3 = self.resblock3(x2)
        x4 = self.resblock4(x3)



        # x1 = self.dub1(x1)
        # x2 = self.dub2(x2)
        # x3 = self.dub3(x3)
        # x4 = self.dub4(x4)


        # out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.cbam(x4)
        out = self.postconv(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

class Para2outV10_1output(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64, 128, 256, 512], atrous_rates=[6, 12, 18]):
        super(Para2outV10_1output, self).__init__()

        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels[0], 7, 2, padding=3),
                                      nn.BatchNorm2d(out_channels[0]), nn.ReLU(), nn.MaxPool2d(3, 2, 1))



        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                         ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                         ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1], out_channels[2]),
                                         ResBlcok_wo_downsample(out_channels[2], out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2], out_channels[3]),
                                         ResBlcok_wo_downsample(out_channels[3], out_channels[3]))



        self.dub1 = DUBV2(out_channels[0], out_channels[0], 4)
        self.dub2 = DUBV2(out_channels[1], out_channels[1], 4)
        self.dub3 = DUBV2(out_channels[2], out_channels[2], 4)
        self.dub4 = DUBV2(out_channels[3], out_channels[3], 4)


        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.cbam = CBAM(sum(out_channels))

        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3], out_features=1), nn.ReLU())

        self.postconv = nn.Sequential(nn.Conv2d(sum(out_channels), out_channels[3], 1, 1),
                                       nn.BatchNorm2d(out_channels[3]), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))



    def forward(self, x):

        x = self.preconv1(x)
        x1 = self.resblock1(
            x)
        x2 = self.resblock2(x1)
        x3 = self.resblock3(x2)
        x4 = self.resblock4(x3)


        x1 = self.dub1(x1)
        x2 = self.dub2(x2)
        x3 = self.dub3(x3)
        x4 = self.dub4(x4)
        # x = self.aspp(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.cbam(out)
        out = self.postconv(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

class Para2outV11(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV11, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.preconv2 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1_1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock1_2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock1_3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock1_4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]),nn.AdaptiveAvgPool2d((1,1)))

        self.resblock2_1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2_2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock2_3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock2_4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]),nn.AdaptiveAvgPool2d((1,1)))

        self.dub1_1 = DUBV2(out_channels[0],out_channels[0],4)
        self.dub1_2 = DUBV2(out_channels[1],out_channels[1],4)
        self.dub1_3 = DUBV2(out_channels[2],out_channels[2],4)
        self.dub1_4 = DUBV2(out_channels[3],out_channels[3],4)

        self.dub2_1 = DUBV2(out_channels[0],out_channels[0],4)
        self.dub2_2 = DUBV2(out_channels[1],out_channels[1],4)
        self.dub2_3 = DUBV2(out_channels[2],out_channels[2],4)
        self.dub2_4 = DUBV2(out_channels[3],out_channels[3],4)
        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.cbam = CBAM(sum(out_channels))

        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3]*2,out_features=2), nn.ReLU())

        self.postconv1 = nn.Sequential(nn.Conv2d(sum(out_channels),out_channels[3],1,1),nn.BatchNorm2d(out_channels[3]),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

        self.postconv2 = nn.Sequential(nn.Conv2d(sum(out_channels),out_channels[3],1,1),nn.BatchNorm2d(out_channels[3]),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

    def forward(self, x):
        x1, x2 = torch.split(x,1,dim=1)
        x1 = self.preconv1(x1)
        x2 = self.preconv2(x2)

        x1_1 = self.resblock1_1(x1)
        x1_2 = self.resblock1_2(x1_1)
        x1_3 = self.resblock1_3(x1_2)
        x1_4 = self.resblock1_4(x1_3)

        # x = self.aspp(x)

        x1 = x1_4
        # x1 = self.postconv1(x1)

        x2_1 = self.resblock2_1(x2)
        x2_2 = self.resblock2_2(x2_1)
        x2_3 = self.resblock2_3(x2_2)
        x2_4 = self.resblock2_4(x2_3)

        x2 = x2_4
        # x2 = self.postconv2(x2)

        out1 = F.sigmoid(x2)*x1+x1
        out2 = F.sigmoid(x1)*x2+x2

        out = torch.cat([out1, out2], dim=1)

        out = out.view(out.shape[0],-1)

        out = self.linear(out)
        return out




class Para2outV12(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV12, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.preconv2 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1_1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock1_2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock1_3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock1_4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]),nn.AdaptiveAvgPool2d((1,1)))

        self.resblock2_1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2_2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock2_3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock2_4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]),nn.AdaptiveAvgPool2d((1,1)))

        self.dub1_1 = DUBV2(out_channels[0],out_channels[0],4)
        self.dub1_2 = DUBV2(out_channels[1],out_channels[1],4)
        self.dub1_3 = DUBV2(out_channels[2],out_channels[2],4)
        self.dub1_4 = DUBV2(out_channels[3],out_channels[3],4)

        self.dub2_1 = DUBV2(out_channels[0],out_channels[0],4)
        self.dub2_2 = DUBV2(out_channels[1],out_channels[1],4)
        self.dub2_3 = DUBV2(out_channels[2],out_channels[2],4)
        self.dub2_4 = DUBV2(out_channels[3],out_channels[3],4)
        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.cbam = CBAM(sum(out_channels))

        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3]*2,out_features=2), nn.ReLU())

        self.postconv1 = nn.Sequential(nn.Conv2d(sum(out_channels),out_channels[3],1,1),nn.BatchNorm2d(out_channels[3]),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

        self.postconv2 = nn.Sequential(nn.Conv2d(sum(out_channels),out_channels[3],1,1),nn.BatchNorm2d(out_channels[3]),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

    def forward(self, x):
        x1, x2 = torch.split(x,1,dim=1)
        x1 = self.preconv1(x1)
        x2 = self.preconv2(x2)

        x1_1 = self.resblock1_1(x1)
        x1_2 = self.resblock1_2(x1_1)
        x1_3 = self.resblock1_3(x1_2)
        x1_4 = self.resblock1_4(x1_3)

        # x = self.aspp(x)

        x1 = x1_4
        # x1 = self.postconv1(x1)

        x2_1 = self.resblock2_1(x2)
        x2_2 = self.resblock2_2(x2_1)
        x2_3 = self.resblock2_3(x2_2)
        x2_4 = self.resblock2_4(x2_3)

        x2 = x2_4
        # x2 = self.postconv2(x2)

        # out1 = F.sigmoid(x2)*x1+x1
        # out2 = F.sigmoid(x1)*x2+x2

        out = torch.cat([x1, x2], dim=1)

        out = out.view(out.shape[0],-1)

        out = self.linear(out)
        return out

class Para2outV13(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV13, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.preconv2 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1_1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock1_2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock1_3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock1_4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]),nn.AdaptiveAvgPool2d((1,1)))

        self.resblock2_1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2_2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock2_3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock2_4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]),nn.AdaptiveAvgPool2d((1,1)))

        self.dub1_1 = DUBV2(out_channels[0],out_channels[0],4)
        self.dub1_2 = DUBV2(out_channels[1],out_channels[1],4)
        self.dub1_3 = DUBV2(out_channels[2],out_channels[2],4)
        self.dub1_4 = DUBV2(out_channels[3],out_channels[3],4)

        self.dub2_1 = DUBV2(out_channels[0],out_channels[0],4)
        self.dub2_2 = DUBV2(out_channels[1],out_channels[1],4)
        self.dub2_3 = DUBV2(out_channels[2],out_channels[2],4)
        self.dub2_4 = DUBV2(out_channels[3],out_channels[3],4)
        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.cbam = CBAM(sum(out_channels))

        self.multihead1 = nn.MultiheadAttention(embed_dim=512,num_heads=8)
        self.multihead2 = nn.MultiheadAttention(embed_dim=512,num_heads=8)


        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3]*2,out_features=2), nn.ReLU())

        self.postconv1 = nn.Sequential(nn.Conv2d(sum(out_channels),out_channels[3],1,1),nn.BatchNorm2d(out_channels[3]),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

        self.postconv2 = nn.Sequential(nn.Conv2d(sum(out_channels),out_channels[3],1,1),nn.BatchNorm2d(out_channels[3]),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

    def forward(self, x):
        x1, x2 = torch.split(x,1,dim=1)
        x1 = self.preconv1(x1)
        x2 = self.preconv2(x2)

        x1_1 = self.resblock1_1(x1)
        x1_2 = self.resblock1_2(x1_1)
        x1_3 = self.resblock1_3(x1_2)
        x1_4 = self.resblock1_4(x1_3)

        # x = self.aspp(x)

        x1 = x1_4
        # x1 = self.postconv1(x1)

        x2_1 = self.resblock2_1(x2)
        x2_2 = self.resblock2_2(x2_1)
        x2_3 = self.resblock2_3(x2_2)
        x2_4 = self.resblock2_4(x2_3)

        x2 = x2_4

        x1 = (x1.view(x1.shape[0], -1)).unsqueeze(0) #1*N*512
        x2 = (x2.view(x2.shape[0],-1)).unsqueeze(0)  #1*N*512

        x12,_ = self.multihead1(x2,x2,x1)
        x21,_ = self.multihead2(x1,x1,x2)

        x12 = x12.squeeze(0)
        x21 = x21.squeeze(0)

        # x2 = self.postconv2(x2)

        # out1 = F.sigmoid(x2)*x1+x1
        # out2 = F.sigmoid(x1)*x2+x2

        out = torch.cat([x12, x21], dim=1)

        # out = out.view(out.shape[0],-1)

        out = self.linear(out)
        return out

class Para2outV14(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64,128,256,512], atrous_rates=[6,12,18]):
        super(Para2outV14, self).__init__()


        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.preconv2 = nn.Sequential(nn.Conv2d(in_channels,out_channels[0],7,2,padding=3),nn.BatchNorm2d(out_channels[0]),nn.ReLU(),nn.MaxPool2d(3,2,1))

        self.resblock1_1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock1_2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock1_3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock1_4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]))

        self.resblock2_1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                       ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2_2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                       ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock2_3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1],out_channels[2]),
                                       ResBlcok_wo_downsample(out_channels[2],out_channels[2]))

        self.resblock2_4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2],out_channels[3]),
                                       ResBlcok_wo_downsample(out_channels[3],out_channels[3]))

        self.dub1_1 = DUBV2(out_channels[0],out_channels[0],4)
        self.dub1_2 = DUBV2(out_channels[1],out_channels[1],4)
        self.dub1_3 = DUBV2(out_channels[2],out_channels[2],4)
        self.dub1_4 = DUBV2(out_channels[3],out_channels[3],4)

        self.dub2_1 = DUBV2(out_channels[0],out_channels[0],4)
        self.dub2_2 = DUBV2(out_channels[1],out_channels[1],4)
        self.dub2_3 = DUBV2(out_channels[2],out_channels[2],4)
        self.dub2_4 = DUBV2(out_channels[3],out_channels[3],4)
        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.cbam = CBAM(2*sum(out_channels))


        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3]*2,out_features=2), nn.ReLU())

        self.postconv = nn.Sequential(nn.Conv2d(2*sum(out_channels),2*out_channels[3],1,1),nn.BatchNorm2d(2*out_channels[3]),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

        # self.postconv2 = nn.Sequential(nn.Conv2d(sum(out_channels),out_channels[3],1,1),nn.BatchNorm2d(out_channels[3]),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))

    def forward(self, x):
        x1, x2 = torch.split(x,1,dim=1)
        x1 = self.preconv1(x1)
        x2 = self.preconv2(x2)

        x1_1 = self.resblock1_1(x1)
        x1_2 = self.resblock1_2(x1_1)
        x1_3 = self.resblock1_3(x1_2)
        x1_4 = self.resblock1_4(x1_3)

        x1_1 = self.dub1_1(x1_1)
        x1_2 = self.dub1_2(x1_2)
        x1_3 = self.dub1_3(x1_3)
        x1_4 = self.dub1_4(x1_4)
        # x = self.aspp(x)


        # x1 = self.postconv1(x1)

        x2_1 = self.resblock2_1(x2)
        x2_2 = self.resblock2_2(x2_1)
        x2_3 = self.resblock2_3(x2_2)
        x2_4 = self.resblock2_4(x2_3)

        x2_1 = self.dub2_1(x2_1)
        x2_2 = self.dub2_2(x2_2)
        x2_3 = self.dub2_3(x2_3)
        x2_4 = self.dub2_4(x2_4)

        out = torch.cat([x1_1,x1_2,x1_3,x1_4,x2_1,x2_2,x2_3,x2_4],dim=1)
        out = self.cbam(out)
        out = self.postconv(out)

        out = out.view(out.shape[0],-1)

        out = self.linear(out)
        return out

class Para2outV15(nn.Module):
    def __init__(self, in_channels=1, out_channels=[64, 128, 256, 512], atrous_rates=[6, 12, 18]):
        super(Para2outV15, self).__init__()

        self.preconv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels[0], 7, 2, padding=3),
                                      nn.BatchNorm2d(out_channels[0]), nn.ReLU(), nn.MaxPool2d(3, 2, 1))



        self.resblock1 = nn.Sequential(ResBlcok_wo_downsample(out_channels[0], out_channels[0]),
                                         ResBlcok_wo_downsample(out_channels[0], out_channels[0]))

        self.resblock2 = nn.Sequential(ResBlcok_w_downsample(out_channels[0], out_channels[1]),
                                         ResBlcok_wo_downsample(out_channels[1], out_channels[1]))

        self.resblock3 = nn.Sequential(ResBlcok_w_downsample(out_channels[1], out_channels[2]),
                                         ResBlcok_wo_downsample(out_channels[2], out_channels[2]))

        self.resblock4 = nn.Sequential(ResBlcok_w_downsample(out_channels[2], out_channels[3]),
                                         ResBlcok_wo_downsample(out_channels[3], out_channels[3]))



        self.dub1 = DUBV2(out_channels[0], out_channels[0], 4)
        self.dub2 = DUBV2(out_channels[1], out_channels[1], 4)
        self.dub3 = DUBV2(out_channels[2], out_channels[2], 4)
        self.dub4 = DUBV2(out_channels[3], out_channels[3], 4)

        self.cbam1 =CBAM(out_channels[0])
        self.cbam2 = CBAM(out_channels[1])
        self.cbam3 = CBAM(out_channels[2])
        self.cbam4 = CBAM(out_channels[3])

        # self.aspp = nn.Sequential(ASPP(out_channels[3],out_channels[3],atrous_rates=atrous_rates), nn.AdaptiveAvgPool2d((1,1)))
        self.cbam5 = CBAM(sum(out_channels))

        self.linear = nn.Sequential(nn.Linear(in_features=out_channels[3], out_features=2), nn.ReLU())

        self.postconv = nn.Sequential(nn.Conv2d(sum(out_channels), out_channels[3], 1, 1),
                                       nn.BatchNorm2d(out_channels[3]), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))



    def forward(self, x):

        x = self.preconv1(x)
        x1 = self.resblock1(
            x)
        x2 = self.resblock2(x1)
        x3 = self.resblock3(x2)
        x4 = self.resblock4(x3)


        x1 = self.cbam1(self.dub1(x1))
        x2 = self.cbam2(self.dub2(x2))
        x3 = self.cbam3(self.dub3(x3))
        x4 = self.cbam4(self.dub4(x4))
        # x = self.aspp(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.cbam5(out)
        out = self.postconv(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out
