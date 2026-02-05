import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size1, kernel_size2, stride1, stride2):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size1, padding= 1, stride=stride1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size2, padding= 1, stride=stride2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


# Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size1, kernel_size2, stride1, stride2):
        super(DecoderBlock, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size1, padding= 1,
                                        stride=stride1,output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size2, padding= 1, stride=stride2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.convT(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


# Bottleneck block
class Bottleneck(nn.Module):
    def __init__(self, filters):
        super(Bottleneck, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(27 * 50, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.reshape = nn.Unflatten(1, (1, 16, 16))  # Reshaping to 16x16

    def forward(self, x):
        batch_size = x.size(0)
        # For the first channel
        c = self.flatten(x[:, 0, :, :])
        c = F.relu(self.fc1(c))
        c = F.relu(self.fc2(c))
        c = F.relu(self.fc3(c))
        c = self.reshape(c)

        # Concatenate other channels
        for k in range(1, x.size(1)):
            c1 = self.flatten(x[:, k, :, :])
            c1 = F.relu(self.fc1(c1))
            c1 = F.relu(self.fc2(c1))
            c1 = F.relu(self.fc3(c1))
            c1 = self.reshape(c1)
            c = torch.cat([c, c1], dim=1)  # Concatenate along the channel axis

        return c


# PINet model
class PINet(nn.Module):
    def __init__(self):
        super(PINet, self).__init__()
        self.encoder1 = EncoderBlock(1, 16, kernel_size1=(3, 3), kernel_size2=(3, 3), stride1=(2, 1), stride2=1)
        self.encoder2 = EncoderBlock(16, 32, kernel_size1=(3, 3), kernel_size2=(3, 3), stride1=(2, 1), stride2=1)
        self.encoder3 = EncoderBlock(32, 64, kernel_size1=(3, 3), kernel_size2=(3, 3), stride1=(2, 1), stride2=1)
        self.encoder4 = EncoderBlock(64, 64, kernel_size1=(3, 3), kernel_size2=(3, 3), stride1=(2, 1), stride2=1)
        self.encoder5 = EncoderBlock(64, 128, kernel_size1=(3, 3), kernel_size2=(3, 3), stride1=(2, 2), stride2=1)
        self.encoder6 = EncoderBlock(128, 128, kernel_size1=(3, 3), kernel_size2=(3, 3), stride1=(2, 2), stride2=1)

        self.bottleneck = Bottleneck(64)

        self.decoder1 = DecoderBlock(128, 128, kernel_size1=(3, 3), kernel_size2=(3, 3), stride1=2, stride2=1)
        self.decoder2 = DecoderBlock(128, 64, kernel_size1=(3, 3), kernel_size2=(3, 3), stride1=2, stride2=1)
        self.decoder3 = DecoderBlock(64, 32, kernel_size1=(3, 3), kernel_size2=(3, 3), stride1=2, stride2=1)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.encoder1(x)  # 850*199 -> 425*199
        x = self.encoder2(x)  # 425*199 -> 213*199
        x = self.encoder3(x)  # 213*199 -> 107*199
        x = self.encoder4(x)  # 107*199 -> 54*100
        x = self.encoder5(x)  # 54*100 -> 27*54
        x = self.encoder6(x)  # 27*54 -> bottleneck

        # Bottleneck
        x = self.bottleneck(x)  # bottleneck

        # Decoder
        x = self.decoder1(x)  # 16*16 -> 32*32
        x = self.decoder2(x)  # 32*32 -> 64*64
        x = self.decoder3(x)  # 64*64 -> 128*128

        # Final output layer
        output = F.relu(self.final_conv(x))
        return output



