import torch
import torch.nn as nn
import torch.nn.init as init



def initialize_conv2d_xavier_uniform(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Apply Xavier/Glorot uniform initialization to convolutional layers
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


def initialize_conv2d_xavier_uniformV2(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Apply Xavier/Glorot uniform initialization to convolutional layers
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    for n in model.modules():
        if isinstance(n, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)