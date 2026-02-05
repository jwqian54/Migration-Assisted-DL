import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from torchvision import transforms
from torch.autograd import Variable

# You may need to install the `pytorch-msssim` library for SSIM calculation
# You can install it using: pip install pytorch-msssim
from pytorch_msssim import ssim

class HybridLoss(nn.Module):
    def __init__(self, alpha_ssim=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha_ssim

    def forward(self, predicted, target):
        # Calculate Mean Squared Error (MSE)
        mse_loss = F.mse_loss(predicted, target)

        # Calculate Structural Similarity Index (SSIM)
        ssim_loss = 1 - ssim(predicted, target, data_range=target.max() - target.min(), nonnegative_ssim=True)

        # Combine MSE and SSIM with a weighted factor alpha
        hybrid_loss = (1 - self.alpha) * mse_loss + self.alpha * ssim_loss

        return hybrid_loss, ssim_loss, mse_loss

class HybridLossV2(nn.Module):
    def __init__(self, alpha_ssim=0.5):
        super(HybridLossV2, self).__init__()
        self.alpha = alpha_ssim
        self.bceloss = nn.BCELoss()
    def forward(self, predicted, target):
        # Calculate Mean Squared Error (MSE)
        bce_loss = self.bceloss(predicted, target)

        # Calculate Structural Similarity Index (SSIM)
        ssim_loss = 1 - ssim(predicted, target, data_range=target.max() - target.min(), nonnegative_ssim=True)
        # ssim_loss = bce_loss
        # Combine MSE and SSIM with a weighted factor alpha
        hybrid_loss = (1 - self.alpha) * bce_loss + self.alpha * ssim_loss

        return hybrid_loss, ssim_loss, bce_loss

class HybridLoss_w_crop(nn.Module):
    def __init__(self, alpha_ssim=0.5, crop_threshold =5):
        super(HybridLoss_w_crop, self).__init__()
        self.alpha = alpha_ssim
        self.threshold = crop_threshold

    def forward(self, predicted, target, crop_info):
        # Calculate Mean Squared Error (MSE)
        mse_loss = F.mse_loss(predicted, target)

        # Calculate Structural Similarity Index (SSIM)

        ssim_loss = image_crop_within_batch(predicted, target, crop_info, threshold=self.threshold)
        # ssim_loss = 1 - ssim(predicted, target, data_range=target.max() - target.min(), nonnegative_ssim=True)

        # Combine MSE and SSIM with a weighted factor alpha
        hybrid_loss = (1 - self.alpha) * mse_loss + self.alpha * ssim_loss

        return hybrid_loss, ssim_loss, mse_loss

def image_crop_within_batch(y_pred,y_target, crop_info, threshold = 10):
    ssim_loss = []
    for ii_sample in range(len(y_target)):

        crop_target = y_target[ii_sample,:,crop_info[ii_sample,0]-threshold:crop_info[ii_sample, 1]+threshold, crop_info[ii_sample,2]-threshold:crop_info[ii_sample,3]+threshold].unsqueeze(0)

        crop_pred = y_pred[ii_sample,:,crop_info[ii_sample,0]-threshold:crop_info[ii_sample, 1]+threshold, crop_info[ii_sample,2]-threshold:crop_info[ii_sample,3]+threshold].unsqueeze(0)

        dssim = 1-ssim(crop_pred, crop_target, data_range=crop_target.max()-crop_target.min(), nonnegative_ssim=True)
        ssim_loss.append(dssim)

    ssim_loss_stack = torch.stack(ssim_loss)


    return torch.mean(ssim_loss_stack, dim =0)



def norm_tensor(input, range_min=0.0, range_max=1.0):
    input_min = torch.min(input, dim=(2,3),keepdim = True)
    input_max = torch.max(input, dim = (2,3), keepdim= True)

    eps_value= 1e-5

    if input_max-input_min<=eps_value:
        scaled_input = (input - input_min) / eps_value * (range_max - range_min) + range_min
    else:
        scaled_input = (input - input_min) / (input_max - input_min) * (range_max - range_min) + range_min
    print("shape of input",input.shape)
    print("shape of scaled input", scaled_input.shape)
    return scaled_input