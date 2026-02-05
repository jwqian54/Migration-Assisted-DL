import numpy as np
from sklearn import preprocessing

import scipy.io as sio
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from skimage.exposure import rescale_intensity
from sklearn.preprocessing import MinMaxScaler
import random

import matplotlib.pyplot as plt


# this function returns tensor directoly
def load_mat_file_pytorch(filename, prefix, scale=True):
    data = []
    mat_content = sio.loadmat(filename)
    IMG_SIZEX = 128
    IMG_SIZEY = 128
    # idx=0
    for key in mat_content:
        if key.startswith(prefix):
            tmp = mat_content[key]
            if scale == True:
                tmp = preprocessing.scale(tmp).astype(np.float32)
            tmp = torch.from_numpy(tmp.reshape(1, 1, IMG_SIZEX, IMG_SIZEY))
            # make sure the tensor is with float32
            tmp = tmp.to(torch.float32)
            data.append(tmp)
            # idx += 1
    data = torch.cat(data, dim=0)
    return data


def load_mat_file_pytorch_1dim(filename, prefix, scale=False):
    data = []
    mat_content = sio.loadmat(filename)
    # idx=0
    for key in mat_content:
        if key.startswith(prefix):
            tmp = mat_content[key]
            if scale == True:
                tmp = preprocessing.scale(tmp).astype(np.float32)
            tmp = torch.from_numpy(tmp.reshape(1, 1))
            # make sure the tensor is with float32
            tmp = tmp.to(torch.float32)
            data.append(tmp)
            # idx += 1
    data = torch.cat(data, dim=0)
    return data


def load_1stStage_label(mat_file_path, per_vector_name, train_index_name, test_index_name, suffix='', shift_aug=False,
                        Regression=False):
    mat_data = sio.loadmat(mat_file_path)

    effective_per = mat_data[per_vector_name]
    effective_per = torch.tensor(effective_per)

    train_index = torch.tensor(train_index_name.astype(np.int16))
    test_index = torch.tensor(test_index_name.astype(np.int16))

    train_data = effective_per[0, (train_index - 1).long()]
    test_data = effective_per[0, (test_index - 1).long()]

    if Regression == True:
        train_data_label = torch.tensor(train_data) / torch.max(train_data)
        test_data_label = torch.tensor(test_data) / torch.max(train_data)
    else:

        vector_values = torch.range(2.5, 10.5, 0.5)
        value_to_class = {val.item(): idx for idx, val in enumerate(vector_values)}

        train_data_label = torch.tensor([value_to_class[val.item()] for val in train_data])

        test_data_label = torch.tensor([value_to_class[val.item()] for val in test_data])

    if shift_aug == True:
        train_data_label = train_data_label.repeat(60)

    if Regression == False:
        return (train_data_label).to(torch.long), (test_data_label).to(torch.long)
    else:
        return train_data_label.unsqueeze(1), test_data_label.unsqueeze(1)


def load_1stStage_value(mat_file_path, per_vector_name, train_index_name, test_index_name, suffix='', shift_aug=False,
                        Regression=False):
    mat_data = sio.loadmat(mat_file_path)

    effective_per = mat_data[per_vector_name]
    effective_per = torch.tensor(effective_per)

    train_index = torch.tensor(train_index_name.astype(np.int16))
    test_index = torch.tensor(test_index_name.astype(np.int16))

    train_data = effective_per[0, (train_index - 1).long()]
    test_data = effective_per[0, (test_index - 1).long()]

    if Regression == True:
        train_data_label = torch.tensor(train_data)
        test_data_label = torch.tensor(test_data)
    else:

        vector_values = torch.range(2.5, 10.5, 0.5)
        value_to_class = {val.item(): idx for idx, val in enumerate(vector_values)}

        train_data_label = torch.tensor([value_to_class[val.item()] for val in train_data])

        test_data_label = torch.tensor([value_to_class[val.item()] for val in test_data])

    if shift_aug == True:
        train_data_label = train_data_label.repeat(60)

    if Regression == False:
        return (train_data_label).to(torch.long), (test_data_label).to(torch.long)
    else:
        return train_data_label.unsqueeze(1), test_data_label.unsqueeze(1)


def load_2ndStage_decayvalue(mat_file_path, decay_prefix, decay_suffix, index_digit, train_index_name, test_index_name,
                             shift_aug=False, Regression=False):
    mat_data = sio.loadmat(mat_file_path)

    train_index = torch.tensor(train_index_name.astype(np.int16))
    test_index = torch.tensor(test_index_name.astype(np.int16))

    train_decay = []
    test_decay = []

    for tree_index in train_index:
        tree_index_padded = str(tree_index.item()).zfill(index_digit)
        variable_name = f"{decay_prefix}{tree_index_padded}{decay_suffix}"
        tmp = mat_data[variable_name]

        tmp = torch.from_numpy(tmp.reshape(1, 1))
        # make sure the tensor is with float32
        tmp = tmp.to(torch.float32)

        train_decay.append(tmp)

    for tree_index in test_index:
        tree_index_padded = str(tree_index.item()).zfill(index_digit)
        variable_name = f"{decay_prefix}{tree_index_padded}{decay_suffix}"
        tmp = mat_data[variable_name]
        tmp = torch.from_numpy(tmp.reshape(1, 1))
        # make sure the tensor is with float32
        tmp = tmp.to(torch.float32)
        test_decay.append(tmp)

    if shift_aug == True:
        train_decay = train_decay * 60

        # idx += 1
    train_decay = torch.cat(train_decay, dim=0)
    test_decay = torch.cat(test_decay, dim=0)

    return train_decay, test_decay


def resize_array(array, new_shape):
    zoom_factors = (new_shape[0] / array.shape[0], new_shape[1] / array.shape[1])
    resized_array = zoom(array, zoom_factors, order=1)  # Linear interpolation (order=1)
    return resized_array


def shift_columns(array,
                  num_shifts):  # if num_shifts is positive,for example num_shifts = 1, then the last 1 column becomes the first column
    # if num_shifts is negative, for example, num_shifts = -1, then the first 1 column becomes the last
    return np.roll(array, shift=num_shifts, axis=1)  # Shift along columns (axis 1)


def load_1stStage_Bscan(mat_file_path, Bscan_prefix, Bscan_suffix, index_digit, train_index_name, test_index_name,
                        shift_aug=False):
    mat_data = sio.loadmat(mat_file_path)

    train_index = torch.tensor(train_index_name.astype(np.int16))
    test_index = torch.tensor(test_index_name.astype(np.int16))

    train_Bscan = []
    test_Bscan = []

    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(60)

    for ii_aug in iteration_range:
        for tree_index in train_index:
            tree_index_padded = str(tree_index.item()).zfill(index_digit)
            variable_name = f"{Bscan_prefix}{tree_index_padded}{Bscan_suffix}"
            tmp = mat_data[variable_name]
            tmp_resize = resize_array(tmp, (128, 60))
            tmp_shifted = shift_columns(tmp_resize, -ii_aug)
            tmp_resizeback = resize_array(tmp_shifted, (128, 128))
            train_Bscan.append(tmp_resizeback)

    # note that the test dataset doesnt need augmentation.
    for tree_index in test_index:
        tree_index_padded = str(tree_index.item()).zfill(index_digit)
        variable_name = f"{Bscan_prefix}{tree_index_padded}{Bscan_suffix}"
        tmp = mat_data[variable_name]
        test_Bscan.append(tmp)

    train_Bscan = torch.tensor(np.array(train_Bscan)).unsqueeze(1)
    test_Bscan = torch.tensor(np.array(test_Bscan)).unsqueeze(1)

    return train_Bscan, test_Bscan


# it should be noted that, for the loading of the migration image, the test_index is removed becasue the generation of the migration image in the test dataset is in a separate file with a predicted permittivity value.
def load_2ndStage_Migration(mat_file_path, Bscan_prefix, Bscan_suffix, index_digit, train_index_name, shift_aug=False, preprocessing= 0):
    mat_data = sio.loadmat(mat_file_path)

    train_index = torch.tensor(train_index_name.astype(np.int16))

    train_Bscan = []

    scaler1 = MinMaxScaler(feature_range=(-1,1))
    scaler2 = MinMaxScaler(feature_range=(0,1))

    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(60)

    for ii_aug in iteration_range:
        for tree_index in train_index:
            tree_index_padded = str(tree_index.item()).zfill(index_digit)
            variable_name = f"{Bscan_prefix}{tree_index_padded}{Bscan_suffix}"
            tmp = mat_data[variable_name]
            if preprocessing == 1:

                tmp = scaler1.fit_transform(tmp)
                tmp = tmp**2
                tmp = scaler2.fit_transform(tmp)



            tmp_resize = resize_array(tmp, (128, 60))
            tmp_shifted = shift_columns(tmp_resize, -ii_aug)
            tmp_resizeback = resize_array(tmp_shifted, (128, 128))
            train_Bscan.append(tmp_resizeback)

    # note that the test dataset doesnt need augmentation.
    train_Bscan = torch.tensor(np.array(train_Bscan)).unsqueeze(1)

    return train_Bscan



def load_Measure_Migration_rotateaug(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)
    inputsize = 182
    outputsize = 128
    start_index = int(1 + (inputsize - outputsize) / 2) - 1;  # -1 means python starts index from 0
    stop_index = int(start_index + outputsize);
    to_pil = transforms.ToPILImage()

    data_list = []

    if shift_aug == True:
        num_rotate = 180
    else:
        num_rotate = 1

    for ii_rotate in range(num_rotate):
        rotation_angle = -360 / num_rotate * (
            ii_rotate)  # positive rotation angle means counter clockwise, in this case, we are doing clockwise, therefore, it should be negative .

        for ii_tree in tree_index:

            for ii_decay in range(1,5):

                if prefix.startswith("Migra"):

                    variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}_{polar_index}_svd{svd_index}"
                elif prefix.startswith("Geo"):
                    variable_name = f"{prefix}_tree{ii_tree}"
                elif prefix.startswith("Per"):
                    variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"

                tmp = mat_data[variable_name]
                tmp = torch.from_numpy(tmp).unsqueeze(0)  # transfer to 1*182*182
                tmp_rotate = TF.rotate(tmp, rotation_angle, interpolation=InterpolationMode.NEAREST)
                tmp_rotate = tmp_rotate[:, start_index:stop_index, start_index:stop_index].unsqueeze(0) # transfer to 1*1*128*128

                data_list.append(tmp_rotate)

    data_list = torch.cat(data_list,dim=0)


                # img = to_pil(tmp_rotate)
                #
                # plt.imshow(img)
                # plt.savefig(f"test_decay{ii_decay + 1}_rotate_{360 / num_rotate * ii_rotate}.png")
                # plt.close()

    return data_list

def load_Measure_Migration_rotateaug_Tree(mat_file_path, prefix, tree_index, decay_index, polar_index, shift_aug=True, RandomStatus = False):
    mat_data = sio.loadmat(mat_file_path)
    inputsize = 182
    outputsize = 128
    start_index = int(1 + (inputsize - outputsize) / 2) - 1;  # -1 means python starts index from 0
    stop_index = int(start_index + outputsize);
    to_pil = transforms.ToPILImage()

    data_list = []

    if shift_aug == True:
        num_rotate = 60
    else:
        num_rotate = 1

    for ii_rotate in range(num_rotate):
        rotation_angle = -360 / num_rotate * (
            ii_rotate)  # positive rotation angle means counter clockwise, in this case, we are doing clockwise, therefore, it should be negative .

        for ii_sample in range(len(tree_index)):
            ii_tree = tree_index[ii_sample]
            ii_decay = decay_index[ii_sample]

            if prefix.startswith("Migra"):
                variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}_{polar_index}"
            elif prefix.startswith("Geo"):
                variable_name = f"{prefix}_tree{ii_tree}"
            elif prefix.startswith("Per"):
                variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"

            tmp = mat_data[variable_name]



            tmp = torch.from_numpy(tmp).unsqueeze(0)  # transfer to 1*182*182
            tmp_rotate = TF.rotate(tmp, rotation_angle, interpolation=InterpolationMode.NEAREST)
            tmp_rotate = tmp_rotate[:, start_index:stop_index, start_index:stop_index].unsqueeze(0) # transfer to 1*1*128*128

            data_list.append(tmp_rotate)



    if RandomStatus==True:

        random.seed(54)
        random.shuffle(data_list)

                # img = to_pil(tmp_rotate)
    data_list = torch.cat(data_list, dim=0)
                # plt.imshow(img)
                # plt.savefig(f"test_decay{ii_decay + 1}_rotate_{360 / num_rotate * ii_rotate}.png")
                # plt.close()

    return data_list




def load_Measure_Migration_rotateaugV3(mat_file_path, prefix, tree_index, polar_index, shift_aug=True, preprocessing=0):
    mat_data = sio.loadmat(mat_file_path)
    inputsize = 182
    outputsize = 128
    start_index = int(1 + (inputsize - outputsize) / 2) - 1;  # -1 means python starts index from 0
    stop_index = int(start_index + outputsize);
    to_pil = transforms.ToPILImage()

    scaler1 = MinMaxScaler(feature_range=(-1,1))
    scaler2 = MinMaxScaler(feature_range=(0,1))


    data_list = []

    if shift_aug == True:
        num_rotate = 180
    else:
        num_rotate = 1

    for ii_rotate in range(num_rotate):
        rotation_angle = -360 / num_rotate * (
            ii_rotate)  # positive rotation angle means counter clockwise, in this case, we are doing clockwise, therefore, it should be negative .

        for ii_tree in tree_index:

            for ii_decay in range(1,5):

                if prefix.startswith("Migra"):

                    variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}_{polar_index}"
                elif prefix.startswith("Geo"):
                    variable_name = f"{prefix}_tree{ii_tree}"
                elif prefix.startswith("Per"):
                    variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"

                tmp = mat_data[variable_name]

                if preprocessing==1:
                    tmp = scaler1.fit_transform(tmp)
                    tmp = tmp ** 2
                    tmp = scaler2.fit_transform(tmp)

                tmp = torch.from_numpy(tmp).unsqueeze(0)  # transfer to 1*182*182
                tmp_rotate = TF.rotate(tmp, rotation_angle, interpolation=InterpolationMode.NEAREST)
                tmp_rotate = tmp_rotate[:, start_index:stop_index, start_index:stop_index].unsqueeze(0) # transfer to 1*1*128*128

                data_list.append(tmp_rotate)

    data_list = torch.cat(data_list,dim=0)


                # img = to_pil(tmp_rotate)
                #
                # plt.imshow(img)
                # plt.savefig(f"test_decay{ii_decay + 1}_rotate_{360 / num_rotate * ii_rotate}.png")
                # plt.close()

    return data_list

# in V4, we add the random shift to the migration data to do the augmentation.
def load_Measure_Migration_rotateaugV4(mat_file_path,prefix, tree_index, polar_index, shift_aug=True, random_shift = True):
    mat_data = sio.loadmat(mat_file_path)

    inputsize = 182
    outputsize = 128
    start_index = int(1 + (inputsize - outputsize) / 2) - 1;  # -1 means python starts index from 0
    stop_index = int(start_index + outputsize);
    to_pil = transforms.ToPILImage()

    data_list = []
    max_shift = 25

    if random_shift ==True:

        np.random.seed(54)
        shifts = np.random.randint(-max_shift, max_shift+1, size = (20,2))

    else:
        shifts = np.random.randint(-max_shift, max_shift+1, size = (1,2))




    if shift_aug == True:
        num_rotate = 30
    else:
        num_rotate = 1

    for ii_shift, (x_shift, y_shift) in enumerate (shifts):

        if ii_shift == 0 :

            x_shift = 0
            y_shift = 0

        pad_x = (max(0, x_shift), max(0, -x_shift))
        pad_y = (max(0, y_shift), max(0, -y_shift))



        for ii_rotate in range(num_rotate):
            rotation_angle = -360 / num_rotate * (
                ii_rotate)  # positive rotation angle means counter clockwise, in this case, we are doing clockwise, therefore, it should be negative .

            for ii_tree in tree_index:

                for ii_decay in range(1,5):

                    if prefix.startswith("Migra"):

                        variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}_{polar_index}"
                    elif prefix.startswith("Geo"):
                        variable_name = f"{prefix}_tree{ii_tree}"
                    elif prefix.startswith("Per"):
                        variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"

                    tmp = mat_data[variable_name]
                    tmp = torch.from_numpy(tmp).unsqueeze(0)
                    tmp_rotate = TF.rotate(tmp, rotation_angle, interpolation=InterpolationMode.NEAREST)

                    #crop first
                    tmp_rotate = tmp_rotate[:, start_index:stop_index, start_index:stop_index].squeeze(0)
                    tmp_rotate = tmp_rotate.numpy()

                    #apply padding and crop
                    tmp_padded = np.pad(tmp_rotate, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])), mode='constant', constant_values=0)



                    tmp_shifted = tmp_padded[:128, :128]



                    tmp_shifted = (tmp_shifted-tmp_shifted.min())/(tmp_shifted.max()-tmp_shifted.min())

                    tmp_shifted = torch.from_numpy(tmp_shifted).unsqueeze(0).unsqueeze(0)
                     # transfer to 1*182*182


                    data_list.append(tmp_shifted)

    data_list = torch.cat(data_list,dim=0)


                # # img = to_pil(tmp_rotate)
                # #
                # plt.imshow(tmp_rotate)
                # plt.savefig(f"test_decay{ii_decay + 1}_rotate_{360 / num_rotate * ii_rotate}.png")
                # plt.close()

    return data_list


def load_Measure_DecayValue(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)


    data_list = []

    if shift_aug == True:
        num_aug = 180
    else:
        num_aug = 1

    for ii_tree in tree_index:

        for ii_decay in range(1,5):

            if prefix.startswith("Per"):
                variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"

            tmp = mat_data[variable_name]
            tmp = torch.from_numpy(tmp.reshape(1, 1))
            # make sure the tensor is with float32
            tmp = tmp.to(torch.float32)
            data_list.append(tmp)

    data_list = data_list * num_aug

            # idx += 1
    data_list = torch.cat(data_list, dim=0)

    return data_list

def load_Measure_DecayValue_Tree(mat_file_path, prefix, tree_index, decay_list,  polar_index,svd_index, shift_aug=True, randomstatus=False):
    mat_data = sio.loadmat(mat_file_path)

    data_list = []

    if shift_aug == True:
        num_aug = 60
    else:
        num_aug = 1

    for ii_tree in tree_index:

        for ii_decay in decay_list:

            if prefix.startswith("Per"):
                variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"

            if prefix.startswith("Er"):
                variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar_index}"

            tmp = mat_data[variable_name]
            tmp = torch.from_numpy(tmp.reshape(1, 1))
            # make sure the tensor is with float32
            tmp = tmp.to(torch.float32)
            data_list.append(tmp)


    data_list = data_list * num_aug

    if randomstatus==True:
        random.seed(54)
        random.shuffle(data_list)

    data_list = torch.cat(data_list, dim=0)

    return data_list

def load_Measure_DecayValue_TreeV2(mat_file_path, prefix, tree_index, decay_list,  polar_index,svd_index, shift_aug=True, randomstatus=False):
    mat_data = sio.loadmat(mat_file_path)

    data_list = []

    if shift_aug == True:
        num_aug = 60
    else:
        num_aug = 1

    for ii_sample in range(len(tree_index)):
        ii_tree = tree_index[ii_sample]
        ii_decay = decay_list[ii_sample]

        if prefix.startswith("Per"):
            variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"

        elif prefix.startswith("Er"):
            variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar_index}"

        tmp = mat_data[variable_name]
        tmp = torch.from_numpy(tmp.reshape(1, 1))
        # make sure the tensor is with float32
        tmp = tmp.to(torch.float32)
        data_list.append(tmp)


    data_list = data_list * num_aug

    if randomstatus==True:
        random.seed(54)
        random.shuffle(data_list)

    data_list = torch.cat(data_list, dim=0)

    return data_list

def load_Measure_DecayValue_2decay(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)
    decay_value = [1,4]

    data_list = []

    if shift_aug == True:
        num_aug = 180
    else:
        num_aug = 1

    for ii_tree in tree_index:

        for ii_decay in decay_value:

            if prefix.startswith("Per"):
                variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"


            tmp = mat_data[variable_name]
            tmp = torch.from_numpy(tmp.reshape(1, 1))
            # make sure the tensor is with float32
            tmp = tmp.to(torch.float32)
            data_list.append(tmp)

    data_list = data_list * num_aug

            # idx += 1
    data_list = torch.cat(data_list, dim=0)

    return data_list

#Mix is to split the dataset not accoording to the tree index, but randomly selected from all datasample
def load_Measure_DecayValue_Mix(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)


    data_list = []

    for ii_tree in tree_index:

        for ii_decay in range(1,5):

            if prefix.startswith("Per"):
                variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"

            tmp = mat_data[variable_name]
            tmp = torch.from_numpy(tmp.reshape(1, 1))
            # make sure the tensor is with float32
            tmp = tmp.to(torch.float32)
            data_list.append(tmp)

    random.seed(42)
    random.shuffle(data_list)
    split_index = int(len(data_list) * 0.8)
    train_list = data_list[:split_index]
    test_list = data_list[split_index:]


    if shift_aug == True:
        num_aug = 180
    else:
        num_aug = 1


    train_list = train_list * num_aug


    train_list = torch.cat(train_list, dim =0)
    test_list = torch.cat(test_list, dim=0)
            # idx += 1


    return train_list, test_list

def load_Measure_DecayValue_Bucket_Mix(mat_file_path, prefix, geo_list, polar_index, shift_aug=True, Value= True):
    mat_data = sio.loadmat(mat_file_path)


    data_list = []
    train_data_list = []
    test_data_list = []
    inputsize = 182
    outputsize = 128
    start_index = int(1 + (inputsize - outputsize) / 2) - 1;  # -1 means python starts index from 0
    stop_index = int(start_index + outputsize);
    to_pil = transforms.ToPILImage()

    for ii_decay in range(1, 5):
        for ii_geo in geo_list:

            for ii_pos in range (1,11):

                variable_name = f"{prefix}_{ii_geo}_decay{ii_decay}_pos{ii_pos}_{polar_index}"

                if variable_name in mat_data:

                    tmp = mat_data[variable_name]

                    if Value==True:
                        tmp = torch.from_numpy(tmp.reshape(1, 1))
                        # make sure the tensor is with float32
                        tmp = tmp.to(torch.float32)


                    else:
                        tmp = torch.from_numpy(tmp).unsqueeze(0)  # transfer to 1*182*182
                    data_list.append(tmp)

    random.seed(42)
    random.shuffle(data_list)
    split_index = int(len(data_list) * 0.8)
    train_list = data_list[:split_index]
    test_list = data_list[split_index:]


    if shift_aug == True:
        num_aug = 60
    else:
        num_aug = 1

    if Value ==True:
        train_data_list = train_list * num_aug

        test_data_list = test_list
    else:
        for ii_aug in range(num_aug):
            rotation_angle = -360 / num_aug * (
                ii_aug)  # positive rotation angle means counter clockwise, in this case, we are doing clockwise, therefore, it should be negative .
            for tmp in train_list:
                tmp_rotate = TF.rotate(tmp, rotation_angle, interpolation=InterpolationMode.NEAREST)
                tmp_rotate = tmp_rotate[:, start_index:stop_index, start_index:stop_index].unsqueeze(
                    0)  # transfer to 1*1*128*128

                train_data_list.append(tmp_rotate)


        for tmp in test_list:
            tmp = tmp[:, start_index:stop_index, start_index:stop_index].unsqueeze(0)
            test_data_list.append(tmp)


    train_data_list = torch.cat(train_data_list, dim =0)
    test_data_list = torch.cat(test_data_list, dim=0)
            # idx += 1


    return train_data_list, test_data_list


def load_Measure_DecayValue_Mix_2decay(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)

    decay_value = [1,4]

    data_list = []

    for ii_tree in tree_index:

        for ii_decay in decay_value:

            if prefix.startswith("Per"):
                variable_name = f"{prefix}_tree{ii_tree}_decay{ii_decay}"

            tmp = mat_data[variable_name]
            tmp = torch.from_numpy(tmp.reshape(1, 1))
            # make sure the tensor is with float32
            tmp = tmp.to(torch.float32)
            data_list.append(tmp)

    random.seed(42)
    random.shuffle(data_list)
    split_index = int(len(data_list) * 0.8)
    train_list = data_list[:split_index]
    test_list = data_list[split_index:]


    if shift_aug == True:
        num_aug = 180
    else:
        num_aug = 1


    train_list = train_list * num_aug


    train_list = torch.cat(train_list, dim =0)
    test_list = torch.cat(test_list, dim=0)
            # idx += 1


    return train_list, test_list

# v2 is to read Kaixuan's selected_er.mat
def load_Measure_Migration_rotateaugV2(mat_file_path, prefix, tree_index,er_selected_mat, er_selected_variable,migratype, shift_aug=True):
    mat_data = sio.loadmat(mat_file_path)
    inputsize = 182
    outputsize = 128
    start_index = int(1 + (inputsize - outputsize) / 2) - 1;  # -1 means python starts index from 0
    stop_index = int(start_index + outputsize);
    to_pil = transforms.ToPILImage()

    mat_data2 = sio.loadmat(er_selected_mat)
    migra_select = mat_data2[er_selected_variable]
    # migra_select is 32*4*6, where 32 refers to tree index, 4 refers to decay index, and 6 refers to svdindex, polar, ssim_index, ssim_er, entropy_index, entropy_er, respectively.

    data_list = []

    if shift_aug == True:
        num_rotate = 180
    else:
        num_rotate = 1

    for ii_rotate in range(num_rotate):
        rotation_angle = -360 / num_rotate * (
            ii_rotate)  # positive rotation angle means counter clockwise, in this case, we are doing clockwise, therefore, it should be negative .

        for ii_tree in tree_index:

            for ii_decay in range(1,5):


                svd = f"{migra_select[ii_tree-1, ii_decay-1, 0].squeeze()}"#the value of svd is from 0 to 3

                if migra_select[ii_tree-1, ii_decay-1, 1].squeeze()==2:
                    polar = "s22_ra"
                elif migra_select[ii_tree-1, ii_decay-1, 1].squeeze()==1:
                    polar = "s11_ra"

                # polar = f"{migra_select[ii_tree-1, ii_decay-1, 1].squeeze()}_ra"

                if migratype.startswith("ssim"):
                    er_index = f"{migra_select[ii_tree-1,ii_decay-1, 2].squeeze()}"
                elif migratype.startswith("entropy"):
                    er_index = f"{migra_select[ii_tree-1,ii_decay-1, 5].squeeze()}"



                variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar}_00001"

                tmp = mat_data[variable_name] #it's a 4d vector, 4*10*182*182
                tmp = tmp[int(float(svd)),int(float(er_index)-1),:,:]
                tmp = torch.from_numpy(tmp).unsqueeze(0)  # transfer to 1*182*182
                tmp_rotate = TF.rotate(tmp, rotation_angle, interpolation=InterpolationMode.NEAREST)


                tmp_rotate = tmp_rotate[:, start_index:stop_index, start_index:stop_index]
                tmp_rotate = tmp_rotate.squeeze(0).numpy()
                tmp_rotate = rescale_intensity(tmp_rotate,in_range='image',out_range=(0,1))
                tmp_rotate = torch.from_numpy(tmp_rotate).unsqueeze(0).unsqueeze(0) # transfer to 1*1*128*128

                data_list.append(tmp_rotate)

    data_list = torch.cat(data_list,dim=0)

    return data_list

def load_Measure_Bscan_shiftaug(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)


    data_list = []

    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(180)

    for ii_aug in iteration_range:

        for ii_tree in tree_index:

            for ii_decay in range(1,5):

                variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar_index}"

                tmp = mat_data[variable_name]
                tmp_resize = resize_array(tmp, (128, 180))
                tmp_shifted = shift_columns(tmp_resize, -ii_aug)
                tmp_resizeback = resize_array(tmp_shifted, (128, 128))
                tmp_resizeback = torch.from_numpy(tmp_resizeback).unsqueeze(0).unsqueeze(0)
                data_list.append(tmp_resizeback)



    data_list = torch.cat(data_list,dim=0)


                # img = to_pil(tmp_rotate)
                #
                # plt.imshow(img)
                # plt.savefig(f"test_decay{ii_decay + 1}_rotate_{360 / num_rotate * ii_rotate}.png")
                # plt.close()

    return data_list

def load_Measure_Bscan_shiftaug_Tree(mat_file_path, prefix, tree_index, decay_list, polar_index, pinet=False, shift_aug=True, RandomStatus=False):
    mat_data = sio.loadmat(mat_file_path)
    data_list = []

    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(60)

    for ii_aug in iteration_range:

        for ii_tree in tree_index:

            for ii_decay in decay_list:

                variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar_index}"

                tmp = mat_data[variable_name]
                tmp_resize = resize_array(tmp, (128, 60))
                tmp_shifted = shift_columns(tmp_resize, -ii_aug)
                tmp_resizeback = resize_array(tmp_shifted, (128, 128))
                if pinet==True:
                    tmp_resizeback = resize_array(tmp_resizeback, (1700, 199))
                tmp_resizeback = torch.from_numpy(tmp_resizeback).unsqueeze(0).unsqueeze(0)
                data_list.append(tmp_resizeback)

    if RandomStatus==True:

        random.seed(54)
        random.shuffle(data_list)


    data_list = torch.cat(data_list,dim=0)


                # img = to_pil(tmp_rotate)
                #
                # plt.imshow(img)
                # plt.savefig(f"test_decay{ii_decay + 1}_rotate_{360 / num_rotate * ii_rotate}.png")
                # plt.close()

    return data_list


def load_Measure_Bscan_shiftaug_TreeV2(mat_file_path, prefix, tree_index, decay_list, polar_index, svd_index,pinet =False, shift_aug=True, randomstatus=False):
    mat_data = sio.loadmat(mat_file_path)
    data_list = []

    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(60)

    for ii_aug in iteration_range:

        for ii_sample in range(len(tree_index)):
            ii_tree = tree_index[ii_sample]
            ii_decay = decay_list[ii_sample]

            variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar_index}"

            tmp = mat_data[variable_name]
            tmp_resize = resize_array(tmp, (128, 60))
            tmp_shifted = shift_columns(tmp_resize, -ii_aug)
            tmp_resizeback = resize_array(tmp_shifted, (128, 128))
            if pinet == True:
                tmp_resizeback = resize_array(tmp_resizeback, (1700, 199))
            tmp_resizeback = torch.from_numpy(tmp_resizeback).unsqueeze(0).unsqueeze(0)
            data_list.append(tmp_resizeback)

    if randomstatus==True:

        random.seed(54)
        random.shuffle(data_list)


    data_list = torch.cat(data_list,dim=0)


                # img = to_pil(tmp_rotate)
                #
                # plt.imshow(img)
                # plt.savefig(f"test_decay{ii_decay + 1}_rotate_{360 / num_rotate * ii_rotate}.png")
                # plt.close()

    return data_list

def load_Measure_Bscan_shiftaug_2decay(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)

    decay_value = [1,4]

    data_list = []

    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(180)

    for ii_aug in iteration_range:

        for ii_tree in tree_index:

            for ii_decay in decay_value:

                variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar_index}"

                tmp = mat_data[variable_name]
                tmp_resize = resize_array(tmp, (128, 180))
                tmp_shifted = shift_columns(tmp_resize, -ii_aug)
                tmp_resizeback = resize_array(tmp_shifted, (128, 128))
                tmp_resizeback = torch.from_numpy(tmp_resizeback).unsqueeze(0).unsqueeze(0)
                data_list.append(tmp_resizeback)



    data_list = torch.cat(data_list,dim=0)


                # img = to_pil(tmp_rotate)
                #
                # plt.imshow(img)
                # plt.savefig(f"test_decay{ii_decay + 1}_rotate_{360 / num_rotate * ii_rotate}.png")
                # plt.close()

    return data_list

def load_Measure_Bscan_shiftaug_Mix(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)


    data_list = []

    train_data = []
    test_data = []

    for ii_tree in tree_index:

        for ii_decay in range(1, 5):
            variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar_index}"
            tmp = mat_data[variable_name]

            data_list.append(tmp)

    random.seed(42)
    random.shuffle(data_list)
    split_index = int(len(data_list)*0.8)
    train_list = data_list[:split_index]
    test_list = data_list[split_index:]


    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(180)

    for ii_aug in iteration_range:

        for tmp in train_list:

            tmp_resize = resize_array(tmp, (128, 180))
            tmp_shifted = shift_columns(tmp_resize, -ii_aug)
            tmp_resizeback = resize_array(tmp_shifted, (128, 128))
            tmp_resizeback = torch.from_numpy(tmp_resizeback).unsqueeze(0).unsqueeze(0)
            train_data.append(tmp_resizeback)


    for tmp in test_list:
        tmp_resize = torch.from_numpy(tmp).unsqueeze(0).unsqueeze(0)
        test_data.append(tmp_resize)


    train_data = torch.cat(train_data, dim=0)
    test_data = torch.cat(test_data, dim=0)


    return train_data, test_data


def load_Measure_Bscan_shiftaug_Bucket_Mix(mat_file_path, prefix, geo_list, polar_index, shift_aug=True, PinNet=False, ReturnName = False):
    mat_data = sio.loadmat(mat_file_path)


    data_list = []
    data_name_list = []
    train_data = []
    test_data = []
    for ii_decay in range(1, 5):
        for ii_geo in geo_list:
            for ii_pos in range(1,11):

                variable_name = f"{prefix}_{ii_geo}_decay{ii_decay}_pos{ii_pos}_{polar_index}"

                identifier = f"{ii_geo}_decay{ii_decay}_pos{ii_pos}_{polar_index}"

                if variable_name in mat_data:
                    tmp = mat_data[variable_name]

                    data_name_list.append(identifier)
                    data_list.append(tmp)

    random.seed(42)
    random.shuffle(data_name_list)

    random.seed(42)
    random.shuffle(data_list)
    split_index = int(len(data_list)*0.8)
    train_list = data_list[:split_index]
    test_list = data_list[split_index:]

    train_name_list = data_name_list[:split_index]
    test_name_list = data_name_list[split_index:]


    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(60)

    for ii_aug in iteration_range:

        for tmp in train_list:

            tmp_resize = resize_array(tmp, (128, 60))
            tmp_shifted = shift_columns(tmp_resize, -ii_aug)
            tmp_resizeback = resize_array(tmp_shifted, (128, 128))

            if PinNet==True:
                tmp_resizeback = resize_array(tmp_resizeback, (1700, 199))
            tmp_resizeback = torch.from_numpy(tmp_resizeback).unsqueeze(0).unsqueeze(0)
            train_data.append(tmp_resizeback)


    for tmp in test_list:
        if PinNet == True:
            tmp= resize_array(tmp, (1700, 199))
        tmp_resize = torch.from_numpy(tmp).unsqueeze(0).unsqueeze(0)
        test_data.append(tmp_resize)


    train_data = torch.cat(train_data, dim=0)
    test_data = torch.cat(test_data, dim=0)

    if ReturnName==False:
        return train_data, test_data
    else:
        return train_data, test_data, train_name_list,test_name_list

def load_Measure_Bscan_Sensitivity(mat_file_path, input_prefix,output_prefix):
    mat_data = sio.loadmat(mat_file_path)

    data_list = []
    data_output = []
    input_data = []
    output_data = []

    if input_prefix in mat_data:
        tmp_data = mat_data[input_prefix]
        data_list.append(tmp_data)

    if output_prefix in mat_data:

        tmp_output = mat_data[output_prefix]
        data_output.append(tmp_output)

    for tmp in data_list:
        tmp_resize = torch.from_numpy(tmp).unsqueeze(1)
        input_data.append(tmp_resize)

    for tmp in data_output:
        tmp_output = torch.from_numpy(tmp)
        output_data.append(tmp_output)


    input_data = torch.cat(input_data, dim=0)

    output_data = torch.cat(output_data,dim=0)


    return input_data, output_data




def load_Measure_Bscan_shiftaug_Mix_2decay(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)


    data_list = []

    train_data = []
    test_data = []

    decay_value = [1,4]

    for ii_tree in tree_index:

        for ii_decay in decay_value:
            variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar_index}"
            tmp = mat_data[variable_name]

            data_list.append(tmp)

    random.seed(42)
    random.shuffle(data_list)
    split_index = int(len(data_list)*0.8)
    train_list = data_list[:split_index]
    test_list = data_list[split_index:]


    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(180)

    for ii_aug in iteration_range:

        for tmp in train_list:

            tmp_resize = resize_array(tmp, (128, 180))
            tmp_shifted = shift_columns(tmp_resize, -ii_aug)
            tmp_resizeback = resize_array(tmp_shifted, (128, 128))
            tmp_resizeback = torch.from_numpy(tmp_resizeback).unsqueeze(0).unsqueeze(0)
            train_data.append(tmp_resizeback)


    for tmp in test_list:
        tmp_resize = torch.from_numpy(tmp).unsqueeze(0).unsqueeze(0)
        test_data.append(tmp_resize)


    train_data = torch.cat(train_data, dim=0)
    test_data = torch.cat(test_data, dim=0)


    return train_data, test_data

def load_Measure_Bscan_shiftaug_PINet(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug=True, mode=0):
    mat_data = sio.loadmat(mat_file_path)


    data_list = []

    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(180)

    for ii_aug in iteration_range:

        for ii_tree in tree_index:

            for ii_decay in range(1,5):

                variable_name = f"{prefix}_Tree{ii_tree}_decay{ii_decay}_{polar_index}"

                tmp = mat_data[variable_name]
                tmp_resize = resize_array(tmp, (128, 180))
                tmp_shifted = shift_columns(tmp_resize, -ii_aug)
                tmp_resizeback = resize_array(tmp_shifted, (128, 128))
                tmp_resizeback_resize = resize_array(tmp_resizeback, (1700, 199))
                tmp_resizeback_resize = torch.from_numpy(tmp_resizeback_resize).unsqueeze(0).unsqueeze(0)
                data_list.append(tmp_resizeback_resize)



    data_list = torch.cat(data_list,dim=0)


                # img = to_pil(tmp_rotate)
                #
                # plt.imshow(img)
                # plt.savefig(f"test_decay{ii_decay + 1}_rotate_{360 / num_rotate * ii_rotate}.png")
                # plt.close()

    return data_list

# mat_file_path = "D:\OneDrive - Nanyang Technological University\measurement\Antenna Outside Dataset\Control_test_v2\JW_dataset_circular_SVD_migration_test_v2/Dataset_Migration_1_17.mat"
# prefix = "MigraGray_ssim21"
# tree_index = [1]
# decay_index = 4
# polar_index = "s11_ra"
# svd_index = 2
# shift_aug = True
#
# load_Measure_Migration_rotateaug(mat_file_path, prefix, tree_index, decay_index, polar_index, svd_index, shift_aug)


def load_2ndStage_PiNet(mat_file_path, Bscan_prefix, Bscan_suffix, index_digit, train_index_name, shift_aug=False):
    mat_data = sio.loadmat(mat_file_path)

    train_index = torch.tensor(train_index_name.astype(np.int16))

    train_Bscan = []

    if shift_aug == False:
        iteration_range = range(1)
    else:
        iteration_range = range(60)

    for ii_aug in iteration_range:
        for tree_index in train_index:
            tree_index_padded = str(tree_index.item()).zfill(index_digit)
            variable_name = f"{Bscan_prefix}{tree_index_padded}{Bscan_suffix}"
            tmp = mat_data[variable_name]
            tmp_resize = resize_array(tmp, (128, 60))
            tmp_shifted = shift_columns(tmp_resize, -ii_aug)
            tmp_resizeback = resize_array(tmp_shifted, (128, 128))
            tmp_resizeback_resize = resize_array(tmp_resizeback, (1700, 199))
            train_Bscan.append(tmp_resizeback_resize)

    # note that the test dataset doesnt need augmentation.
    train_Bscan = torch.tensor(np.array(train_Bscan)).unsqueeze(1)

    return train_Bscan


def load_mat_file_pytorch_treeindex(filepath, prefix, train_indice, test_indice, sample_per_tree, scale=True,
                                    method='scale'):
    train_data = []
    test_data = []
    mat_content = sio.loadmat(filepath)
    IMG_SIZEX = 128
    IMG_SIZEY = 128
    # idx=0
    for tree_index in train_indice + test_indice:

        for sample in range(1, sample_per_tree + 1):
            sample_padded = str(sample).zfill(5)
            variable_name = f"{prefix}_Tree{tree_index}_cavity_{sample_padded}"
            try:
                tmp = mat_content[variable_name]
                if scale == True:
                    if method == 'scale':
                        tmp = preprocessing.scale(tmp).astype(np.float32)
                    elif method == 'normalize':
                        tmp = preprocessing.normalize(tmp).astype(np.float32)
                    else:
                        raise ValueError("Invalid preprocessing method. Choose 'scale' or 'normalize'.")
                tmp = torch.from_numpy(tmp.reshape(1, 1, IMG_SIZEX, IMG_SIZEY))
                tmp = tmp.to(torch.float32)

                if tree_index in train_indice:
                    train_data.append(tmp)
                else:
                    test_data.append(tmp)


            except KeyError:
                pass

    train_data = torch.cat(train_data, dim=0)
    torch.manual_seed(54)
    permutation1 = torch.randperm(train_data.shape[0])
    train_data_random = train_data[permutation1]

    test_data = torch.cat(test_data, dim=0)
    return train_data_random, test_data


def load_mat_file_pytorch_square(filename, prefix, scale=True):
    data = []
    mat_content = sio.loadmat(filename)
    IMG_SIZEX = 128
    IMG_SIZEY = 128
    # idx=0
    for key in mat_content:
        if key.startswith(prefix):
            tmp = mat_content[key]
            if scale == True:
                tmp = preprocessing.scale(tmp).astype(np.float32)
            tmp = torch.from_numpy(tmp.reshape(1, 1, IMG_SIZEX, IMG_SIZEY))
            # make sure the tensor is with float32
            tmp = tmp.to(torch.float32)
            tmp = torch.square(tmp)
            data.append(tmp)
            # idx += 1
    data = torch.cat(data, dim=0)
    return data


def load_crop_info(filename, prefix):
    data = []
    mat_content = sio.loadmat(filename)
    IMG_SIZEX = 128
    IMG_SIZEY = 128
    # idx=0
    for key in mat_content:
        if key.startswith(prefix):
            tmp = mat_content[key]

            tmp = torch.from_numpy(tmp.reshape(1, -1))
            tmp = tmp.to(torch.int8)
            data.append(tmp)
            # idx += 1
    data = torch.cat(data, dim=0)
    return data


def load_crop_info_byindex(filename, prefix, train_index_name, index_digit):
    data = []
    mat_content = sio.loadmat(filename)

    train_index = torch.tensor(train_index_name.astype(np.int16))
    for tree_index in train_index:
        tree_index_padded = str(tree_index.item()).zfill(index_digit)
        variable_name = f"{prefix}{tree_index_padded}"
        tmp = mat_content[variable_name]
        tmp = torch.from_numpy(tmp.reshape(1, -1))
        tmp = tmp.to(torch.int8)
        data.append(tmp)

    data = torch.cat(data, dim=0)
    return data


def load_crop_info_treeindex(filepath, prefix, train_indice, test_indice, sample_per_tree):
    train_data = []
    test_data = []
    mat_content = sio.loadmat(filepath)
    # idx=0
    for tree_index in train_indice + test_indice:

        for sample in range(1, sample_per_tree + 1):
            sample_padded = str(sample).zfill(5)
            variable_name = f"{prefix}_Tree{tree_index}_cavity_{sample_padded}"
            try:
                tmp = mat_content[variable_name]

                tmp = torch.from_numpy(tmp.reshape(1, -1))
                tmp = tmp.to(torch.int8)
                if tree_index in train_indice:
                    train_data.append(tmp)
                else:
                    test_data.append(tmp)
            except KeyError:
                pass

    train_data = torch.cat(train_data, dim=0)
    torch.manual_seed(54)
    permutation1 = torch.randperm(train_data.shape[0])
    train_data_random = train_data[permutation1]

    test_data = torch.cat(test_data, dim=0)
    return train_data_random, test_data


def create_data_loaders(train_x, train_y_true, val_x, val_y_true, batch_size=32):
    train_dataset = TensorDataset(train_x, train_y_true)
    test_dataset = TensorDataset(val_x, val_y_true)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_data_loaders_Sensitivity(x, y, batch_size=32):
    train_dataset = TensorDataset(x, y)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def create_data_loaders_W_crop(train_x, train_y_true, val_x, val_y_true, train_crop, val_crop, batch_size=32,
                               test_batch_size=32):
    train_dataset = TensorDataset(train_x, train_y_true, train_crop)
    test_dataset = TensorDataset(val_x, val_y_true, val_crop)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def create_data_loaders_RefStack(train_x1, train_x2, train_y, val_x1, val_x2, val_y, batch_size=32):
    train_dataset = TensorDataset(train_x1, train_x2, train_y)
    test_dataset = TensorDataset(val_x1, val_x2, val_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader
