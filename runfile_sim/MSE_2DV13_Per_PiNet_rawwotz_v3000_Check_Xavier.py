import os
import sys
# 将项目根 D:\tgrs_open 加入路径，以便找到 model/data_input 等包
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import scale

import scipy.io as sio

from model.pinet import PINet

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_input.matload import load_mat_file_pytorch
from data_input.matload import load_crop_info_byindex
from data_input.matload import create_data_loaders_W_crop


from model_train.train_config import train_hybridloss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from LossFunction.hybridloss import HybridLoss
from pytorch_msssim import ssim
from initialization import initialize_conv2d_xavier_uniform

from data_input.matload import load_2ndStage_PiNet
from data_input.matload import load_1stStage_Bscan

# from LossFunction.hybridloss import hybridloss_mse_ssim
# from LossFunction.hybridloss import mse_loss_cal
# from LossFunction.hybridloss import ssim_loss_cal
#
# from LossFunction.hybridloss import hybridloss_mse_ssim2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # 使用第二块GPU（从0开始）





# Load data
data_folder = "D:/NTU_staff_OneDrive/OneDrive - Nanyang Technological University/Dataset/migration_image_ArbDefect_3D/output_mask/result/Mat/"

spliting_index = sio.loadmat(data_folder+"dataset_split_index")
train_index = np.squeeze(spliting_index["train_index"])
test_index = np.squeeze(spliting_index["test_index"])
num_train = train_index.size
num_test = test_index.size

matfilename_migration_train =  "Input_Bscan"
matfilename_migration_test =  "Input_Bscan"

migration_prefix = "rawwotz_twodefect_norm_"
migration_suffix = ""
index_digit = 5

matfilename_geo = "Output_DefectPerMap"
geo_prefix = "defect_per_norm"
geo_suffix = ""



Data_Augmentation = False
batchsize = 32

model_type = f'2ndStage_MSELoss_2DV13_Per_PiNet_Checked_Xavier_Batch{batchsize}_rawwotz_{num_test+num_train}'
model_folder = '../Exclude_2DV13/'+model_type

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

input_train = load_2ndStage_PiNet(data_folder+matfilename_migration_train,migration_prefix,migration_suffix,index_digit,train_index,shift_aug=Data_Augmentation)

input_val = load_2ndStage_PiNet(data_folder+matfilename_migration_test,migration_prefix,migration_suffix,index_digit,test_index,shift_aug=Data_Augmentation)

output_train,output_val = load_1stStage_Bscan(data_folder+matfilename_geo,geo_prefix,geo_suffix,index_digit,train_index,test_index)


crop_name = "edge_info_v3000"

crop_prefix = "edge_"


alpha_ssim = 0.0
crop_threshold = 5
epoch_first_stage = 20



IMG_SIZEX = 128
IMG_SIZEY = 128

crop_train = load_crop_info_byindex(data_folder+crop_name,crop_prefix,train_index,index_digit)
crop_val = load_crop_info_byindex(data_folder+crop_name, crop_prefix,test_index,index_digit)


train_loader, val_loader = create_data_loaders_W_crop(input_train, output_train, input_val, output_val, crop_train, crop_val, batch_size=batchsize)


# Training
def check_cuda():
    _cuda = False
    if torch.cuda.is_available():
        _cuda = True
    return _cuda


model = PINet()
initialize_conv2d_xavier_uniform(model)

# model.summary()
is_cuda = check_cuda()
if is_cuda:
    model.cuda()

criterion = HybridLoss(alpha_ssim=alpha_ssim)
optimizer = optim.Adam(model.parameters(),lr = 1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2,  verbose=True)
total_epoch = 100


path = '/'
model_path = model_folder + '/model.h5'


training_loss_write, training_mse_write, training_ssim_write,testing_loss_write, testing_mse_write, testing_ssim_write = train_hybridloss(train_loader, val_loader, model, model_folder, total_epoch, criterion, optimizer, scheduler,PlotOption=True)


scio.savemat((model_folder+"/metric.mat"),mdict={'train_loss': training_loss_write,'val_loss':testing_loss_write, 'train_mse': training_mse_write, 'train_ssim_loss': training_ssim_write, 'test_mse': testing_mse_write, 'test_ssim_loss':testing_ssim_write})


# load best performance model
model_load = PINet()
is_cuda = check_cuda()
if is_cuda:
    model_load.cuda()
model_load.load_state_dict(torch.load((model_folder + "/best_fold0.pth")))

model_load.eval()
ssim_values= []
mse_values = []
mae_values = []
output_folder = model_folder+'/interation_figure'
with torch.no_grad():
    # x_val, y_true = val_loader[0]
    # y_pred = model_load(x_val)
    output_y_pred = []
    output_y_true = []
    output_x_val = []

    for i, data in enumerate(val_loader):

        x_val, y_true, crop_val = data
        x_val = x_val.float().to('cuda')
        y_true = y_true.float().to('cuda')
        crop_val = crop_val.to('cuda')

        y_pred = model_load(x_val)


        x_val_np = x_val.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        y_pred_np  = y_pred.cpu().numpy()


        for ii_sample in range(len(x_val_np)):

            y_pred_sample = y_pred[ii_sample].unsqueeze(0)
            y_true_sample = y_true[ii_sample].unsqueeze(0)


            mse_value = F.mse_loss(y_pred_sample, y_true_sample)
            ssim_value = ssim(y_pred_sample, y_true_sample, data_range = y_true_sample.max()-y_true_sample.min(), nonnegative_ssim=True)
            mae_value = F.l1_loss(y_pred_sample, y_true_sample)

            ssim_values.append(ssim_value)
            mse_values.append(mse_value)
            mae_values.append(mae_value)

            output_y_pred.append(y_pred[ii_sample].squeeze())
            output_y_true.append(y_true[ii_sample].squeeze())
            output_x_val.append(x_val[ii_sample].squeeze())
            if i==0:

                # output the variable is for plotting of the figure for paper purpose

                os.makedirs(output_folder, exist_ok=True)
                sample_folder = os.path.join(output_folder, f'batch0_sample_{ii_sample}')
                os.makedirs(sample_folder, exist_ok=True)
                figure_name = os.path.join(sample_folder, f'best.png')

                plt.figure(figsize=(8, 4))
                plt.subplot(1, 3, 1)
                plt.title('Input Image1')
                plt.imshow(x_val_np[ii_sample].squeeze(), cmap='gray')

                plt.subplot(1, 3, 2)
                plt.title('Ground Truth')
                plt.imshow(y_true_np[ii_sample].squeeze(), cmap='gray')

                plt.subplot(1, 3, 3)
                plt.title('Predicted Image')
                plt.imshow(y_pred_np[ii_sample].squeeze(), cmap='gray')
                plt.savefig(figure_name)

                plt.close()

ssim_values_np = torch.stack(ssim_values).cpu().numpy()
mse_values_np = torch.stack(mse_values).cpu().numpy()
mae_values_np = torch.stack(mae_values).cpu().numpy()

output_y_pred_np = torch.stack(output_y_pred).cpu().numpy()
output_y_true_np = torch.stack(output_y_true).cpu().numpy()
output_x_val_np = torch.stack(output_x_val).cpu().numpy()

average_ssim = np.mean(torch.stack(ssim_values).cpu().numpy())
average_mse = np.mean(torch.stack(mse_values).cpu().numpy())
average_mae = np.mean(torch.stack(mae_values).cpu().numpy())

scio.savemat((model_folder+"/best_performance.mat"),mdict={'mse_ave': average_mse,'mse':mse_values_np, 'mae_ave':average_mae,'mae':mae_values_np ,'ssim_ave': average_ssim, 'ssim':ssim_values_np, 'y_pred':output_y_pred_np, 'y_true':output_y_true_np, 'x_val':output_x_val_np})




model_load.load_state_dict(torch.load((model_folder + "/epoch{}.pth".format(total_epoch-1))))

model_load.eval()
ssim_values= []
mse_values = []
mae_values = []
output_folder = model_folder+'/interation_figure'
with torch.no_grad():
    # x_val, y_true = val_loader[0]
    # y_pred = model_load(x_val)
    output_y_pred = []
    output_y_true = []
    output_x_val = []

    for i, data in enumerate(val_loader):

        x_val, y_true, crop_val = data
        x_val = x_val.float().to('cuda')
        y_true = y_true.float().to('cuda')
        crop_val = crop_val.to('cuda')

        y_pred = model_load(x_val)


        x_val_np = x_val.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        y_pred_np  = y_pred.cpu().numpy()


        for ii_sample in range(len(x_val_np)):

            y_pred_sample = y_pred[ii_sample].unsqueeze(0)
            y_true_sample = y_true[ii_sample].unsqueeze(0)


            mse_value = F.mse_loss(y_pred_sample, y_true_sample)
            ssim_value = ssim(y_pred_sample, y_true_sample, data_range = y_true_sample.max()-y_true_sample.min(), nonnegative_ssim=True)
            mae_value = F.l1_loss(y_pred_sample, y_true_sample)

            ssim_values.append(ssim_value)
            mse_values.append(mse_value)
            mae_values.append(mae_value)

            output_y_pred.append(y_pred[ii_sample].squeeze())
            output_y_true.append(y_true[ii_sample].squeeze())
            output_x_val.append(x_val[ii_sample].squeeze())


ssim_values_np = torch.stack(ssim_values).cpu().numpy()
mse_values_np = torch.stack(mse_values).cpu().numpy()
mae_values_np = torch.stack(mae_values).cpu().numpy()

output_y_pred_np = torch.stack(output_y_pred).cpu().numpy()
output_y_true_np = torch.stack(output_y_true).cpu().numpy()
output_x_val_np = torch.stack(output_x_val).cpu().numpy()

average_ssim = np.mean(torch.stack(ssim_values).cpu().numpy())
average_mse = np.mean(torch.stack(mse_values).cpu().numpy())
average_mae = np.mean(torch.stack(mae_values).cpu().numpy())

scio.savemat((model_folder+"/epoch{}_performance.mat".format(total_epoch-1)),mdict={'mse_ave': average_mse,'mse':mse_values_np, 'mae_ave':average_mae,'mae':mae_values_np ,'ssim_ave': average_ssim, 'ssim':ssim_values_np, 'y_pred':output_y_pred_np, 'y_true':output_y_true_np, 'x_val':output_x_val_np})


