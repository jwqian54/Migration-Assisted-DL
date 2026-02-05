import os
import sys
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import scale

import scipy.io as sio

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from loaddata.matload import load_mat_file_pytorch
from loaddata.matload import load_crop_info
from loaddata.matload import create_data_loaders
from loaddata.matload import load_mat_file_pytorch_1dim
from torchvision.models import resnet18

from model_train.train_config import train
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from LossFunction.hybridloss import HybridLoss
from pytorch_msssim import ssim
from initialization import initialize_conv2d_xavier_uniform

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # 使用第二块GPU（从0开始）

# Load data

def evaluate_bestper_model(model, model_folder, val_loader):

    model.eval()

    with torch.no_grad():
        # x_val, y_true = val_loader[0]
        # y_pred = model_load(x_val)
        output_y_pred = []
        output_y_true = []
        output_x_val = []

        for i, data in enumerate(val_loader):

            x_val, y_true = data
            x_val = x_val.float().to('cuda')
            y_true = y_true.float().to('cuda')

            y_pred = model(x_val)

            x_val_np = x_val.cpu().numpy()
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()


            output_x_val.append(x_val)
            output_y_true.append(y_true)
            output_y_pred.append(y_pred)

        output_x_val = torch.cat(output_x_val).cpu().numpy()
        if output_x_val.shape[1]==1:
            output_x_val = np.squeeze(output_x_val,axis=1)

        output_y_true = torch.cat(output_y_true).cpu().numpy()
        output_y_pred = torch.cat(output_y_pred).cpu().numpy()


    scio.savemat((model_folder + "/best_performance_output.mat"),
                 mdict={'x_val': output_x_val, 'y_true': output_y_true, 'y_pred': output_y_pred})


def evaluate_bestper_model_sensitivity(model, model_folder, val_loader,snr):

    model.eval()

    with torch.no_grad():
        # x_val, y_true = val_loader[0]
        # y_pred = model_load(x_val)
        output_y_pred = []
        output_y_true = []
        output_x_val = []

        for i, data in enumerate(val_loader):

            x_val, y_true = data
            x_val = x_val.float().to('cuda')
            y_true = y_true.float().to('cuda')

            y_pred = model(x_val)

            x_val_np = x_val.cpu().numpy()
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()


            output_x_val.append(x_val)
            output_y_true.append(y_true)
            output_y_pred.append(y_pred)

        output_x_val = torch.cat(output_x_val).cpu().numpy()
        if output_x_val.shape[1]==1:
            output_x_val = np.squeeze(output_x_val,axis=1)

        output_y_true = torch.cat(output_y_true).cpu().numpy()
        output_y_pred = torch.cat(output_y_pred).cpu().numpy()


    scio.savemat((model_folder + "/best_performance_SNR"+str(snr)+".mat"),
                 mdict={'x_val_': output_x_val, 'y_true': output_y_true, 'y_pred': output_y_pred})

def evaluate_1stStage_model(model, model_folder, val_loader):

    model.eval()

    with torch.no_grad():
        # x_val, y_true = val_loader[0]
        # y_pred = model_load(x_val)
        output_y_pred = []
        output_y_true = []
        output_x_val = []

        for i, data in enumerate(val_loader):

            x_val, y_true = data
            x_val = x_val.to('cuda')
            y_true = y_true.to('cuda')

            y_pred = model(x_val)

            x_val_np = x_val.cpu().numpy()
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            _, testing_predicted = torch.max(y_pred.data, 1)

            output_x_val.append(x_val)
            output_y_true.append(y_true)
            output_y_pred.append(testing_predicted)

        output_x_val = torch.cat(output_x_val).cpu().numpy()
        output_x_val = np.squeeze(output_x_val,axis=1)

        output_y_true = torch.cat(output_y_true).cpu().numpy()
        output_y_pred = torch.cat(output_y_pred).cpu().numpy()


    scio.savemat((model_folder + "/1stStage_best_performance_output.mat"),
                 mdict={'x_val': output_x_val, 'y_true': output_y_true, 'y_pred': output_y_pred})

