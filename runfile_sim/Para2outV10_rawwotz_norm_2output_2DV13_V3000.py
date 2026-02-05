import os
import sys
# 将项目根 D:\tgrs_open 加入路径，以便直接运行本脚本时能找到 model/data_input 等包
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import scipy.io as scio
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.ResAttention import Para2outV10
from model_train.train_config import train
from initialization import initialize_conv2d_xavier_uniform
from bestmodeleval import evaluate_bestper_model
from data_input.matload import load_2ndStage_decayvalue
from data_input.matload import load_1stStage_value
from data_input.matload import load_1stStage_Bscan
from data_input.matload import create_data_loaders

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # 使用第二块GPU（从0开始）





# Load data（数据目录：与 runfile_sim 同级的 data_sim）
data_folder = os.path.join(_root, "data_sim") + os.sep

# decay的per的normalize的value都没变
matfilename_decay = "Output_DefectPerValue"
decayvalue_prefix = "per_defect_normV4_"
decayvalue_suffix = ''

matfilename_layer = "Output_LayerPerValue"
layerper_vector_name = "er_ssim2_total_norm"

matfilename_Bscan = "Input_Bscan"
Bscan_prefix = "rawwotz_twodefect_norm_"
Bscan_suffx = ""
index_digit = 5

spliting_index = sio.loadmat(data_folder+"dataset_split_index")
train_index = np.squeeze(spliting_index["train_index"])
test_index = np.squeeze(spliting_index["test_index"])

num_train = train_index.size
num_test = test_index.size

alpha_ssim = 0.0
crop_threshold = 5
epoch_first_stage = 20
total_epoch = 100


batchsize = 32


IMG_SIZEX = 128
IMG_SIZEY = 128

Data_Augmentation = True
Regression = True



input_train, input_test = load_1stStage_Bscan(data_folder+matfilename_Bscan,Bscan_prefix,Bscan_suffx,index_digit,train_index,test_index,shift_aug=Data_Augmentation)

output_train_layer, output_test_layer = load_1stStage_value(data_folder+matfilename_layer,layerper_vector_name,train_index, test_index,shift_aug=Data_Augmentation,Regression=Regression)

output_train_decay, output_test_decay = load_2ndStage_decayvalue(data_folder+matfilename_decay,decayvalue_prefix,decayvalue_suffix,index_digit,train_index, test_index,shift_aug=Data_Augmentation,Regression=Regression)

output_train = torch.cat((output_train_layer,output_train_decay),dim=1)

output_test = torch.cat((output_test_layer,output_test_decay),dim=1)



# 默认单折，不再循环
ii_fold = 0

model_type = f'2PerEst_Para2outV10_Batch{batchsize}_rawwotz_Aug{Data_Augmentation}_Regression{Regression}_{num_test+num_train}_2DV13_fold{ii_fold}'
# model_folder = '../Exclude_2DV13/' + model_type
model_folder = '../ThesisCheck/'+model_type
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

train_loader, val_loader = create_data_loaders(input_train, output_train, input_test, output_test, batch_size=batchsize)


# Training
def check_cuda():
    _cuda = False
    if torch.cuda.is_available():
        _cuda = True
    return _cuda


atrous_rates = [6,12,18]
channels = [64,128,256,512]
model = Para2outV10(1,channels,atrous_rates)

initialize_conv2d_xavier_uniform(model)

# model.summary()
is_cuda = check_cuda()
if is_cuda:
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5,  verbose=True)


path = '/'
model_path = model_folder + '/model.h5'


training_loss_write,testing_loss_write = train(train_loader, val_loader, model, model_folder, total_epoch, criterion, optimizer, scheduler,PlotOption=False)


scio.savemat((model_folder+"/metric.mat"),mdict={'train_loss': training_loss_write,'val_loss':testing_loss_write})

# Testing



# load best performance model
model_load = Para2outV10(1,channels,atrous_rates)

is_cuda = check_cuda()
if is_cuda:
    model_load.cuda()
model_load.load_state_dict(torch.load((model_folder + "/best_fold0.pth")))

evaluate_bestper_model(model_load, model_folder, val_loader)



