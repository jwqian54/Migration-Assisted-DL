import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, strides=1):
        super(FeatureExtractionBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,padding=padding,stride=strides)

        self.conv2 = nn.Conv2d(out_channel, out_channel,kernel_size=kernel_size,padding=padding,stride=2)

        self.relu = nn.ReLU()

        self.batch1 = nn.BatchNorm2d(out_channel)
        self.batch2 = nn.BatchNorm2d(out_channel)
    def forward(self,x):

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        return x

class ParameterEstimation(nn.Module):
    def __init__(self, in_channle, out_channels,level = 5, kernel_size=3, padding=1, strides = 1):
        super(ParameterEstimation, self).__init__()

        out_channel1 = int(out_channels/(2**(level-1)))

        self.extract1 = FeatureExtractionBlock(in_channel=in_channle, out_channel=out_channel1, kernel_size=kernel_size, padding=padding, strides=strides)

        self.extract2 = FeatureExtractionBlock(in_channel=out_channel1, out_channel=out_channel1*2, kernel_size=kernel_size, padding=padding, strides=strides)

        self.extract3 = FeatureExtractionBlock(in_channel=out_channel1*2, out_channel=out_channel1*4, kernel_size=kernel_size, padding=padding, strides=strides)

        self.extract4 = FeatureExtractionBlock(in_channel=out_channel1*4, out_channel=out_channel1*8, kernel_size=kernel_size, padding=padding, strides=strides)

        self.extract5 = FeatureExtractionBlock(in_channel=out_channel1*8, out_channel=out_channel1*16, kernel_size=kernel_size, padding=padding, strides=strides)

        self.pool=nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout()
        self.linear1 = nn.Sequential(nn.Flatten(),
                                     nn.Linear(in_features= out_channel1*16, out_features=1))
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.extract1(x)
        x = self.extract2(x)
        x = self.extract3(x)
        x = self.extract4(x)
        x = self.extract5(x)

        x = self.pool(x)

        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)

        return x


