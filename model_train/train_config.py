import torch
import copy
import os
import matplotlib.pyplot as plt
from LossFunction.hybridloss import HybridLoss
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
def train(TrainLoader,TestLoader, model, model_type,num_epoch, criterion, optimizer,schedule,PlotOption = False,return2loss=False):
    # training_acc_write=[]
    training_loss_write=[]
    # testing_acc_write = []
    testing_loss_write = []
    # best_test_acc = 0.6
    best_test_loss = 5
    fold_num =0

    train_medium_loss_write = []
    train_defect_loss_write = []

    test_medium_loss_write = []
    test_defect_loss_write =[]


    for epoch in range(num_epoch):

        running_loss = 0.0
        total_training_sample = 0.0
        loss_train_medium =0.0
        loss_train_defect=0.0
        model.train()
        for i, data in enumerate(TrainLoader):
            # get the inputs; data is a list of [inputs, y_true]
            bscan_train, y_true = data

            bscan_train = bscan_train.float().cuda()
            y_true = y_true.float().cuda()

            y_pred = model(bscan_train)
            loss = criterion(y_pred, y_true)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_training_sample += y_true.size(0)

            running_loss += loss.item()*y_true.size(0)

            if return2loss:
                loss_train_medium += criterion(y_pred[:,0],y_true[:,0]).item()*y_true.size(0)
                loss_train_defect += criterion(y_pred[:,1],y_true[:,1]).item()*y_true.size(0)


            # display epoch performance
            if i % len(TrainLoader) == len(TrainLoader) - 1:

                training_loss = running_loss / total_training_sample

                # training_acc_write.append(training_acc)
                training_loss_write.append(training_loss)

                if return2loss:
                    loss_train_medium = loss_train_medium/total_training_sample
                    loss_train_defect = loss_train_defect/total_training_sample
                    train_medium_loss_write.append(loss_train_medium)
                    train_defect_loss_write.append(loss_train_defect)

        # since we're not training, we don't need to calculate the gradients for our outputs
        if return2loss:
            testing_loss, loss_test_medium, loss_test_defect = test(TestLoader,model,model_type,criterion,epoch, PlotOption=PlotOption, return2loss=True)
            test_medium_loss_write.append(loss_test_medium)
            test_defect_loss_write.append(loss_test_defect)
        else:
            testing_loss = test(TestLoader,model,model_type,criterion,epoch, PlotOption=PlotOption)
        schedule.step(testing_loss)
        print(f'Epoch [{epoch+1}/{num_epoch}] - Learning Rate: {optimizer.param_groups[0]["lr"]}')
        testing_loss_write.append(testing_loss)

        # save best performance model
        if (testing_loss < best_test_loss):
            best_test_loss = testing_loss
            best_epoch = epoch +1
            torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))

        print('[epoch %5d ] training loss: %.5f   test loss: %.5f  best test loss: %.5f ' %
              (epoch + 1, training_loss,  testing_loss, best_test_loss))
        if epoch == num_epoch-1:
            try:
                best_epoch
            except NameError:
                best_epoch_exists = False
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))
                print('best epoch does not exist, save the model of the last epoch,')
                best_test_loss = testing_loss
                print('best epoch:   NaN   best test loss:    %.5f'%(best_test_loss))
            else:
                best_epoch_exists = True
                print('best epoch:   %5d   best test loss:    %.5f'%(best_epoch, best_test_loss))

    if return2loss:

        return training_loss_write, testing_loss_write, train_medium_loss_write, train_defect_loss_write, test_medium_loss_write, test_defect_loss_write
    else:
        return training_loss_write, testing_loss_write


def test(DataLoader, model,model_type, criterion,epoch, PlotOption=False, return2loss=False):
    model.eval()
    total_testing_sample = 0.0
    # testing_acc = 0.0
    testing_loss = 0.0
    loss_medium = 0.0
    loss_defect = 0.0
    with torch.no_grad():
        # for j, data_test in enumerate(test_loader):

        for i, data_test in enumerate(DataLoader):
            bscan_test, y_true = data_test
            bscan_test = bscan_test.float().cuda()
            y_true = y_true.float().cuda()

            # calculate outputs by running images through the network
            outputs_test = model(bscan_test)
            loss2 = criterion(outputs_test, y_true)



            testing_loss += loss2.item() * y_true.size(0)
            total_testing_sample += y_true.size(0)

            if return2loss:

                loss_medium += criterion(outputs_test[:,0],y_true[:,0]).item()*y_true.size(0)
                loss_defect += criterion(outputs_test[:,1],y_true[:,1]).item()*y_true.size(0)

            if (PlotOption == True) & (i<1) :
                x_val_np = bscan_test.cpu().numpy()
                y_true_np = y_true.cpu().numpy()
                y_pred_np = outputs_test.cpu().numpy()

                for ii_sample in range(len(x_val_np)):
                    output_folder = model_type + '/interation_figure'
                    os.makedirs(output_folder, exist_ok=True)
                    sample_folder = os.path.join(output_folder, f'batch0_sample_{ii_sample}')
                    os.makedirs(sample_folder, exist_ok=True)
                    figure_name = os.path.join(sample_folder, f'epoch{epoch}.png')

                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 3, 1)
                    plt.title('Input Image1')
                    plt.imshow(x_val_np[ii_sample].squeeze(), cmap='gray')

                    # plt.subplot(1, 3, 2)
                    # plt.title('Input Image2')
                    # plt.imshow(input_image[:,:,1], cmap='gray')

                    plt.subplot(1, 3, 2)
                    plt.title('Ground Truth')
                    plt.imshow(y_true_np[ii_sample].squeeze(), cmap='gray')

                    plt.subplot(1, 3, 3)
                    plt.title('Predicted Image')
                    plt.imshow(y_pred_np[ii_sample].squeeze(), cmap='gray')
                    plt.savefig(figure_name)

                    plt.close()

            # if PlotOption == True:
            #     ylabel.extend(testing_groundtruth.cpu().numpy())
            #     ypred.extend(testing_predicted.cpu().numpy())

        testing_loss = testing_loss / total_testing_sample
    # if PlotOption==True:
    #     return testing_acc, testing_loss, ylabel, ypred
    # else:
    if return2loss:
        loss_medium = loss_medium/total_testing_sample
        loss_defect = loss_defect/total_testing_sample
        return testing_loss, loss_medium,loss_defect
    else:
        return testing_loss




def train_1stStage(TrainLoader,TestLoader, model, model_type,num_epoch, criterion, optimizer,schedule,fold_num, ResultProtection = False):
    training_acc_write=[]
    training_loss_write=[]
    testing_acc_write = []
    testing_loss_write = []
    best_test_acc = 0.6
    best_test_loss = 1



    for epoch in range(num_epoch):

        running_loss = 0.0
        training_total = 0.0
        num_true = 0.0
        model.train()
        for i, data in enumerate(TrainLoader):
            # get the inputs; data is a list of [inputs, labels]
            bscan_train, labels = data


            bscan_train = bscan_train.float().cuda()
            labels = labels.cuda()

            outputs = model(bscan_train)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _1, training_predicted = torch.max(outputs.data, 1)


            training_total += labels.size(0)
            num_true += (training_predicted == labels).sum().item()

            running_loss += loss.item()

            # display epoch performance
            if i % len(TrainLoader) == len(TrainLoader) - 1:
                training_acc = num_true / training_total
                training_loss = running_loss / len(TrainLoader)

                training_acc_write.append(training_acc)
                training_loss_write.append(training_loss)


        # since we're not training, we don't need to calculate the gradients for our outputs
        testing_acc,testing_loss = test_1stStage(TestLoader,model,criterion,PlotOption=False)
        schedule.step(testing_acc)
        print(f'Epoch [{epoch+1}/{num_epoch}] - Learning Rate: {optimizer.param_groups[0]["lr"]}')
        testing_acc_write.append(testing_acc)
        testing_loss_write.append(testing_loss)

        # save best performance model
        if (testing_acc > best_test_acc):

            if (ResultProtection== True) :
                # only when the testing_acc is less than training_acc, we regard the model is the best performance model
                # this is to avoid the saved model that with testing_acc better than the training acc.
                if (testing_acc-training_acc) < 0.005:
                    best_test_acc = testing_acc
                    best_test_loss = testing_loss
                    best_epoch = epoch + 1
                    torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))

                else:
                    torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}_TestLargerThanTrain.pth".format(fold_num)))

            else:
                best_test_acc = testing_acc
                best_test_loss = testing_loss
                best_epoch = epoch + 1
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))


        elif testing_acc == best_test_acc:
            if testing_loss < best_test_loss:
                best_test_loss = testing_loss
                best_epoch = epoch +1
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))

        print('[epoch %5d ] training loss: %.3f  training acc: %.3f  test loss: %.3f test_acc: %.3f best test loss: %.3f best test acc :%.3f' %
              (epoch + 1, training_loss, training_acc, testing_loss, testing_acc, best_test_loss, best_test_acc))
        if epoch == num_epoch-1:
            try:
                best_epoch
            except NameError:
                best_epoch_exists = False
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))
                print('best epoch does not exist, save the model of the last epoch,')
                best_test_acc = testing_acc
                best_test_loss = testing_loss
                print('best epoch:   NaN   best test acc:   %.3f   best test loss:    %.3f'%(best_test_acc,best_test_loss))
            else:
                best_epoch_exists = True
                print('best epoch:   %5d   best test acc:   %.3f   best test loss:    %.3f'%(best_epoch, best_test_acc,best_test_loss))

    return training_acc_write,training_loss_write,testing_acc_write, testing_loss_write



# Define the test function, PlotOption=True/False determine whether plot confusion matrix or not
def test_1stStage(DataLoader, model,criterion,PlotOption):
    model.eval()
    testing_total = 0.0
    testing_acc = 0.0
    testing_loss = 0.0
    if PlotOption == True:
        ylabel = []
        ypred = []
    with torch.no_grad():
        # for j, data_test in enumerate(test_loader):

        for data_test in DataLoader:
            bscan_test, labels_test = data_test

            bscan_test = bscan_test.float().cuda()
            labels_test = labels_test.cuda()

            # calculate outputs by running images through the network
            outputs_test = model(bscan_test)
            loss2 = criterion(outputs_test, labels_test)
            # the class with the highest energy is what we choose as prediction
            _3, testing_predicted = torch.max(outputs_test.data, 1)


            testing_loss += loss2.item()
            testing_total += labels_test.size(0)
            testing_acc += (testing_predicted == labels_test).sum().item()

            if PlotOption == True:
                ylabel.extend(labels_test.cpu().numpy())
                ypred.extend(testing_predicted.cpu().numpy())
        testing_acc = testing_acc / testing_total
        testing_loss = testing_loss / len(DataLoader)
    if PlotOption==True:
        return testing_acc, testing_loss, ylabel, ypred
    else:
        return testing_acc, testing_loss

def train_hybridloss(TrainLoader,TestLoader, model, model_type,num_epoch, criterion, optimizer,schedule,PlotOption = False, FastSkipping= False):
    # training_acc_write=[]
    training_loss_write=[]
    training_mse_write = []
    training_ssim_write = []

    testing_loss_write = []
    testing_ssim_write = []
    testing_mse_write=[]

    # best_test_acc = 0.6
    best_test_loss = 1
    fold_num =0


    for epoch in range(num_epoch):

        running_loss = 0.0
        running_mse = 0.0
        running_ssim = 0.0
        total_training_sample = 0.0

        model.train()
        for i, data in enumerate(TrainLoader):
            # get the inputs; data is a list of [inputs, y_true]
            bscan_train, y_true, crop_y = data

            bscan_train = bscan_train.float().cuda()
            y_true = y_true.float().cuda()

            crop_y.int().cuda()
            y_pred = model(bscan_train)
            loss,ssim, mse = criterion(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_training_sample += y_true.size(0)
            running_loss += loss.item()*y_true.size(0)

            running_mse +=mse.item()*y_true.size(0)
            running_ssim += ssim.item()*y_true.size(0)

            # display epoch performance
            if i % len(TrainLoader) == len(TrainLoader) - 1:

                training_loss = running_loss / total_training_sample
                training_mse = running_mse / total_training_sample
                training_ssim = running_ssim / total_training_sample
                # training_acc_write.append(training_acc)
                training_loss_write.append(training_loss)
                training_ssim_write.append(training_ssim)
                training_mse_write.append(training_mse)

        # since we're not training, we don't need to calculate the gradients for our outputs
        testing_loss, testing_mse, testing_ssim = test_hybridloss(TestLoader,model,model_type,criterion,epoch, PlotOption=PlotOption)
        schedule.step(testing_loss)
        print(f'Epoch [{epoch+1}/{num_epoch}] - Learning Rate: {optimizer.param_groups[0]["lr"]}')
        testing_loss_write.append(testing_loss)
        testing_mse_write.append(testing_mse)
        testing_ssim_write.append(testing_ssim)


        mse_threshold = 0.020
        if (epoch == 2) & (FastSkipping == True) & (testing_mse > mse_threshold):
            print("  Generating all zero output, terminating training.")
            return training_loss_write,training_mse_write, training_ssim_write, testing_loss_write, testing_mse_write, testing_ssim_write

        # save best performance model
        if (testing_loss < best_test_loss):
            best_test_loss = testing_loss
            best_epoch = epoch +1
            torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))

        print('[epoch %5d ] training loss: %.5f   test loss: %.5f  best test loss: %.5f ' %
              (epoch + 1, training_loss,  testing_loss, best_test_loss))
        if epoch == num_epoch-1:
            try:
                best_epoch
            except NameError:
                best_epoch_exists = False
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))
                print('best epoch does not exist, save the model of the last epoch,')
                best_test_loss = testing_loss
                print('best epoch:   NaN   best test loss:    %.5f'%(best_test_loss))
            else:
                best_epoch_exists = True
                print('best epoch:   %5d   best test loss:    %.5f'%(best_epoch, best_test_loss))
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/epoch{}.pth".format(epoch)))

    return training_loss_write,training_mse_write, training_ssim_write, testing_loss_write, testing_mse_write, testing_ssim_write

def train_hybridlossV2(TrainLoader,TestLoader, model, model_type,num_epoch, criterion, optimizer,schedule,PlotOption = False, FastSkipping= False):
    # training_acc_write=[]
    training_loss_write=[]
    training_mse_write = []
    training_ssim_write = []

    testing_loss_write = []
    testing_ssim_write = []
    testing_mse_write=[]

    # best_test_acc = 0.6
    best_test_loss = 1
    fold_num =0


    for epoch in range(num_epoch):

        running_loss = 0.0
        running_mse = 0.0
        running_ssim = 0.0
        total_training_sample = 0.0

        model.train()
        for i, data in enumerate(TrainLoader):
            # get the inputs; data is a list of [inputs, y_true]
            bscan_train, y_true, crop_y = data

            bscan_train = bscan_train.float().cuda()
            y_true = y_true.float().cuda()

            crop_y.int().cuda()
            y_pred = model(bscan_train)
            loss,ssim, mse = criterion(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_training_sample += y_true.size(0)
            running_loss += loss.item()*y_true.size(0)

            running_mse +=mse.item()*y_true.size(0)
            running_ssim += ssim.item()*y_true.size(0)

            # display epoch performance
            if i % len(TrainLoader) == len(TrainLoader) - 1:

                training_loss = running_loss / total_training_sample
                training_mse = running_mse / total_training_sample
                training_ssim = running_ssim / total_training_sample
                # training_acc_write.append(training_acc)
                training_loss_write.append(training_loss)
                training_ssim_write.append(training_ssim)
                training_mse_write.append(training_mse)

        # since we're not training, we don't need to calculate the gradients for our outputs
        testing_loss, testing_mse, testing_ssim = test_hybridloss(TestLoader,model,model_type,criterion,epoch, PlotOption=PlotOption)
        schedule.step(testing_loss)
        print(f'Epoch [{epoch+1}/{num_epoch}] - Learning Rate: {optimizer.param_groups[0]["lr"]}')
        testing_loss_write.append(testing_loss)
        testing_mse_write.append(testing_mse)
        testing_ssim_write.append(testing_ssim)


        mse_threshold = 0.020
        if (epoch == 2) & (FastSkipping == True) & (testing_mse > mse_threshold):
            print("  Generating all zero output, terminating training.")
            return training_loss_write,training_mse_write, training_ssim_write, testing_loss_write, testing_mse_write, testing_ssim_write

        # save best performance model
        if (testing_loss < best_test_loss):
            best_test_loss = testing_loss
            best_epoch = epoch +1
            torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))

        print('[epoch %5d ] training loss: %.5f   test loss: %.5f  best test loss: %.5f ' %
              (epoch + 1, training_loss,  testing_loss, best_test_loss))
        if epoch == num_epoch-1:
            try:
                best_epoch
            except NameError:
                best_epoch_exists = False
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))
                print('best epoch does not exist, save the model of the last epoch,')
                best_test_loss = testing_loss
                print('best epoch:   NaN   best test loss:    %.5f'%(best_test_loss))
            else:
                best_epoch_exists = True
                print('best epoch:   %5d   best test loss:    %.5f'%(best_epoch, best_test_loss))
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/epoch{}.pth".format(epoch)))

    return training_loss_write,training_mse_write, training_ssim_write, testing_loss_write, testing_mse_write, testing_ssim_write

def test_hybridloss(DataLoader, model,model_type, criterion,epoch, PlotOption=False):
    model.eval()
    total_testing_sample = 0.0
    # testing_acc = 0.0
    testing_loss = 0.0
    testing_mse = 0.0
    testing_ssim = 0.0

    with torch.no_grad():
        # for j, data_test in enumerate(test_loader):

        for i, data_test in enumerate(DataLoader):
            bscan_test, y_true, crop_y = data_test
            bscan_test = bscan_test.float().cuda()
            y_true = y_true.float().cuda()
            crop_y = crop_y.cuda()

            # calculate outputs by running images through the network
            outputs_test = model(bscan_test)
            loss2, ssim, mse = criterion(outputs_test, y_true)

            testing_loss += loss2.item() * y_true.size(0)
            testing_mse += mse.item() * y_true.size(0)
            testing_ssim += ssim.item()* y_true.size(0)

            total_testing_sample += y_true.size(0)

            if (PlotOption == True) & (i<1) :
                x_val_np = bscan_test.cpu().numpy()
                y_true_np = y_true.cpu().numpy()
                y_pred_np = outputs_test.cpu().numpy()

                for ii_sample in range(len(x_val_np)):
                    output_folder = model_type + '/interation_figure'
                    os.makedirs(output_folder, exist_ok=True)
                    sample_folder = os.path.join(output_folder, f'batch0_sample_{ii_sample}')
                    os.makedirs(sample_folder, exist_ok=True)
                    figure_name = os.path.join(sample_folder, f'epoch{epoch}.png')

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


        testing_loss = testing_loss / total_testing_sample
        testing_mse = testing_mse/ total_testing_sample
        testing_ssim = testing_ssim/total_testing_sample

    return testing_loss, testing_mse, testing_ssim

def train_hybridloss_RefStack(TrainLoader,TestLoader, model, model_type,num_epoch, criterion, optimizer,schedule,PlotOption = False):
    # training_acc_write=[]
    training_loss_write=[]
    training_mse_write = []
    training_ssim_write = []

    testing_loss_write = []
    testing_ssim_write = []
    testing_mse_write=[]

    # best_test_acc = 0.6
    best_test_loss = 1
    fold_num =0


    for epoch in range(num_epoch):

        running_loss = 0.0
        running_mse = 0.0
        running_ssim = 0.0
        total_training_sample = 0.0

        model.train()
        for i, data in enumerate(TrainLoader):
            # get the inputs; data is a list of [inputs, y_true]
            bscan_train, bscan_train2, y_true = data

            bscan_train = bscan_train.float().cuda()
            bscan_train2 = bscan_train.float().cuda()
            y_true = y_true.float().cuda()

            y_pred = model(bscan_train, bscan_train2)
            loss,ssim, mse = criterion(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_training_sample += y_true.size(0)
            running_loss += loss.item()*y_true.size(0)

            running_mse +=mse.item()*y_true.size(0)
            running_ssim += ssim.item()*y_true.size(0)

            # display epoch performance
            if i % len(TrainLoader) == len(TrainLoader) - 1:

                training_loss = running_loss / total_training_sample
                training_mse = running_mse / total_training_sample
                training_ssim = running_ssim / total_training_sample
                # training_acc_write.append(training_acc)
                training_loss_write.append(training_loss)
                training_ssim_write.append(training_ssim)
                training_mse_write.append(training_mse)

        # since we're not training, we don't need to calculate the gradients for our outputs
        testing_loss, testing_mse, testing_ssim = test_hybridloss_RefStack(TestLoader,model,model_type,criterion,epoch, PlotOption=PlotOption)
        schedule.step(testing_loss)
        print(f'Epoch [{epoch+1}/{num_epoch}] - Learning Rate: {optimizer.param_groups[0]["lr"]}')
        testing_loss_write.append(testing_loss)
        testing_mse_write.append(testing_mse)
        testing_ssim_write.append(testing_ssim)

        # save best performance model
        if (testing_loss < best_test_loss):
            best_test_loss = testing_loss
            best_epoch = epoch +1
            torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))

        print('[epoch %5d ] training loss: %.5f   test loss: %.5f  best test loss: %.5f ' %
              (epoch + 1, training_loss,  testing_loss, best_test_loss))
        if epoch == num_epoch-1:
            try:
                best_epoch
            except NameError:
                best_epoch_exists = False
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))
                print('best epoch does not exist, save the model of the last epoch,')
                best_test_loss = testing_loss
                print('best epoch:   NaN   best test loss:    %.5f'%(best_test_loss))
            else:
                best_epoch_exists = True
                print('best epoch:   %5d   best test loss:    %.5f'%(best_epoch, best_test_loss))
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/epoch{}.pth".format(epoch)))

    return training_loss_write,training_mse_write, training_ssim_write, testing_loss_write, testing_mse_write, testing_ssim_write

def test_hybridloss_RefStack(DataLoader, model,model_type, criterion,epoch, PlotOption=False):
    model.eval()
    total_testing_sample = 0.0
    # testing_acc = 0.0
    testing_loss = 0.0
    testing_mse = 0.0
    testing_ssim = 0.0

    with torch.no_grad():
        # for j, data_test in enumerate(test_loader):

        for i, data_test in enumerate(DataLoader):
            bscan_test, bscan_test2, y_true = data_test
            bscan_test = bscan_test.float().cuda()
            bscan_test2 = bscan_test2.float().cuda()
            y_true = y_true.float().cuda()


            # calculate outputs by running images through the network
            outputs_test = model(bscan_test,bscan_test2)
            loss2, ssim, mse = criterion(outputs_test, y_true)

            testing_loss += loss2.item() * y_true.size(0)
            testing_mse += mse.item() * y_true.size(0)
            testing_ssim += ssim.item()* y_true.size(0)

            total_testing_sample += y_true.size(0)

            if (PlotOption == True) & (i<1) :
                x_val_np = bscan_test.cpu().numpy()
                x_val_np2= bscan_test2.cpu().numpy()
                y_true_np = y_true.cpu().numpy()
                y_pred_np = outputs_test.cpu().numpy()

                for ii_sample in range(len(x_val_np)):
                    output_folder = model_type + '/interation_figure'
                    os.makedirs(output_folder, exist_ok=True)
                    sample_folder = os.path.join(output_folder, f'batch0_sample_{ii_sample}')
                    os.makedirs(sample_folder, exist_ok=True)
                    figure_name = os.path.join(sample_folder, f'epoch{epoch}.png')

                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 4, 1)
                    plt.title('Input Image1')
                    plt.imshow(x_val_np[ii_sample].squeeze(), cmap='gray')

                    plt.subplot(1, 4, 2)
                    plt.title('Input Image2')
                    plt.imshow(x_val_np2[ii_sample].squeeze(), cmap='gray')

                    plt.subplot(1, 4, 3)
                    plt.title('Ground Truth')
                    plt.imshow(y_true_np[ii_sample].squeeze(), cmap='gray')

                    plt.subplot(1, 4, 4)
                    plt.title('Predicted Image')
                    plt.imshow(y_pred_np[ii_sample].squeeze(), cmap='gray')
                    plt.savefig(figure_name)

                    plt.close()


        testing_loss = testing_loss / total_testing_sample
        testing_mse = testing_mse/ total_testing_sample
        testing_ssim = testing_ssim/total_testing_sample

    return testing_loss, testing_mse, testing_ssim

def train_hybridloss_W_crop(TrainLoader,TestLoader, model, model_type,num_epoch, criterion, optimizer,schedule,PlotOption = False):
    # training_acc_write=[]
    training_loss_write=[]
    training_mse_write = []
    training_ssim_write = []

    testing_loss_write = []
    testing_ssim_write = []
    testing_mse_write=[]

    # best_test_acc = 0.6
    best_test_loss = 1
    fold_num =0


    for epoch in range(num_epoch):

        running_loss = 0.0
        running_mse = 0.0
        running_ssim = 0.0
        total_training_sample = 0.0

        model.train()
        for i, data in enumerate(TrainLoader):
            # get the inputs; data is a list of [inputs, y_true]
            bscan_train, y_true, crop_y = data

            bscan_train = bscan_train.float().cuda()
            y_true = y_true.float().cuda()

            crop_y.int().cuda()
            y_pred = model(bscan_train)
            loss,ssim, mse = criterion(y_pred, y_true,crop_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_training_sample += y_true.size(0)
            running_loss += loss.item()*y_true.size(0)

            running_mse +=mse.item()*y_true.size(0)
            running_ssim += ssim.item()*y_true.size(0)

            # display epoch performance
            if i % len(TrainLoader) == len(TrainLoader) - 1:

                training_loss = running_loss / total_training_sample
                training_mse = running_mse / total_training_sample
                training_ssim = running_ssim / total_training_sample
                # training_acc_write.append(training_acc)
                training_loss_write.append(training_loss)
                training_ssim_write.append(training_ssim)
                training_mse_write.append(training_mse)

        # since we're not training, we don't need to calculate the gradients for our outputs
        testing_loss, testing_mse, testing_ssim = test_W_crop(TestLoader,model,model_type,criterion,epoch, PlotOption=PlotOption)
        schedule.step(testing_loss)
        print(f'Epoch [{epoch+1}/{num_epoch}] - Learning Rate: {optimizer.param_groups[0]["lr"]}')
        testing_loss_write.append(testing_loss)
        testing_mse_write.append(testing_mse)
        testing_ssim_write.append(testing_ssim)

        # save best performance model
        if (testing_loss < best_test_loss):
            best_test_loss = testing_loss
            best_epoch = epoch +1
            torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))

        print('[epoch %5d ] training loss: %.5f   test loss: %.5f  best test loss: %.5f ' %
              (epoch + 1, training_loss,  testing_loss, best_test_loss))
        if epoch == num_epoch-1:
            try:
                best_epoch
            except NameError:
                best_epoch_exists = False
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))
                print('best epoch does not exist, save the model of the last epoch,')
                best_test_loss = testing_loss
                print('best epoch:   NaN   best test loss:    %.5f'%(best_test_loss))
            else:
                best_epoch_exists = True
                print('best epoch:   %5d   best test loss:    %.5f'%(best_epoch, best_test_loss))

    return training_loss_write,training_mse_write, training_ssim_write, testing_loss_write, testing_mse_write, testing_ssim_write

def test_W_crop(DataLoader, model,model_type, criterion,epoch, PlotOption=False):
    model.eval()
    total_testing_sample = 0.0
    # testing_acc = 0.0
    testing_loss = 0.0
    testing_mse = 0.0
    testing_ssim = 0.0

    with torch.no_grad():
        # for j, data_test in enumerate(test_loader):

        for i, data_test in enumerate(DataLoader):
            bscan_test, y_true, crop_y = data_test
            bscan_test = bscan_test.float().cuda()
            y_true = y_true.cuda()
            crop_y = crop_y.cuda()

            # calculate outputs by running images through the network
            outputs_test = model(bscan_test)
            loss2, ssim, mse = criterion(outputs_test, y_true, crop_y)

            testing_loss += loss2.item() * y_true.size(0)
            testing_mse += mse.item() * y_true.size(0)
            testing_ssim += ssim.item()* y_true.size(0)

            total_testing_sample += y_true.size(0)

            if (PlotOption == True) & (i<1) :
                x_val_np = bscan_test.cpu().numpy()
                y_true_np = y_true.cpu().numpy()
                y_pred_np = outputs_test.cpu().numpy()

                for ii_sample in range(len(x_val_np)):
                    output_folder = model_type + '/interation_figure'
                    os.makedirs(output_folder, exist_ok=True)
                    sample_folder = os.path.join(output_folder, f'batch0_sample_{ii_sample}')
                    os.makedirs(sample_folder, exist_ok=True)
                    figure_name = os.path.join(sample_folder, f'epoch{epoch}.png')

                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 3, 1)
                    plt.title('Input Image1')
                    plt.imshow(x_val_np[ii_sample].squeeze(), cmap='gray')

                    # plt.subplot(1, 3, 2)
                    # plt.title('Input Image2')
                    # plt.imshow(input_image[:,:,1], cmap='gray')

                    plt.subplot(1, 3, 2)
                    plt.title('Ground Truth')
                    plt.imshow(y_true_np[ii_sample].squeeze(), cmap='gray')

                    plt.subplot(1, 3, 3)
                    plt.title('Predicted Image')
                    plt.imshow(y_pred_np[ii_sample].squeeze(), cmap='gray')
                    plt.savefig(figure_name)

                    plt.close()

            # if PlotOption == True:
            #     ylabel.extend(testing_groundtruth.cpu().numpy())
            #     ypred.extend(testing_predicted.cpu().numpy())

        testing_loss = testing_loss / total_testing_sample
        testing_mse = testing_mse/ total_testing_sample
        testing_ssim = testing_ssim/total_testing_sample
    # if PlotOption==True:
    #     return testing_acc, testing_loss, ylabel, ypred
    # else:
    return testing_loss, testing_mse, testing_ssim


def train_hybridloss_W_crop_W_TimeVariant(TrainLoader,TestLoader, model, model_type,num_epoch, epoch_first_stage, criterion, optimizer,schedule,PlotOption = False):
    # training_acc_write=[]
    training_loss_write=[]
    training_mse_write = []
    training_ssim_write = []
    training_hybrid_write= []

    testing_loss_write = []
    testing_ssim_write = []
    testing_mse_write=[]
    testing_hybrid_write = []

    # best_test_acc = 0.6
    best_test_loss = 1
    fold_num =0


    for epoch in range(num_epoch):

        running_loss = 0.0
        running_mse = 0.0
        running_ssim = 0.0
        running_hybrid = 0.0
        total_training_sample = 0.0

        model.train()
        for i, data in enumerate(TrainLoader):
            # get the inputs; data is a list of [inputs, y_true]
            bscan_train, y_true, crop_y = data

            bscan_train = bscan_train.float().cuda()
            y_true = y_true.float().cuda()

            crop_y.int().cuda()
            y_pred = model(bscan_train)
            hybrid, ssim, mse = criterion(y_pred, y_true,crop_y)

            if epoch< epoch_first_stage:
                loss = mse
            else:
                loss = hybrid

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_training_sample += y_true.size(0)
            running_loss += loss.item()*y_true.size(0)

            running_mse +=mse.item()*y_true.size(0)
            running_ssim += ssim.item()*y_true.size(0)
            running_hybrid += hybrid.item()*y_true.size(0)
            # display epoch performance
            if i % len(TrainLoader) == len(TrainLoader) - 1:

                training_loss = running_loss / total_training_sample
                training_mse = running_mse / total_training_sample
                training_ssim = running_ssim / total_training_sample
                training_hybrid = running_hybrid/total_training_sample

                # training_acc_write.append(training_acc)
                training_loss_write.append(training_loss)
                training_ssim_write.append(training_ssim)
                training_mse_write.append(training_mse)
                training_hybrid_write.append(training_hybrid)
        # since we're not training, we don't need to calculate the gradients for our outputs
        testing_loss, testing_mse, testing_ssim, testing_hybrid = test_W_crop_W_TimeVariant(TestLoader,model,model_type,criterion,epoch,epoch_first_stage, PlotOption=PlotOption)
        schedule.step(testing_loss)
        print(f'Epoch [{epoch+1}/{num_epoch}] - Learning Rate: {optimizer.param_groups[0]["lr"]}')
        testing_loss_write.append(testing_loss)
        testing_mse_write.append(testing_mse)
        testing_ssim_write.append(testing_ssim)
        testing_hybrid_write.append(testing_hybrid)

        # save best performance model
        if (testing_loss < best_test_loss)& (epoch>epoch_first_stage+1):
            best_test_loss = testing_loss
            best_epoch = epoch +1
            torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))

        print('[epoch %5d ] training loss: %.5f   test loss: %.5f  best test loss: %.5f ' %
              (epoch + 1, training_loss,  testing_loss, best_test_loss))
        if epoch == num_epoch-1:
            try:
                best_epoch
            except NameError:
                best_epoch_exists = False
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/best_fold{}.pth".format(fold_num)))
                print('best epoch does not exist, save the model of the last epoch,')
                best_test_loss = testing_loss
                print('best epoch:   NaN   best test loss:    %.5f'%(best_test_loss))
            else:
                best_epoch_exists = True
                print('best epoch:   %5d   best test loss:    %.5f'%(best_epoch, best_test_loss))
                torch.save(copy.deepcopy(model.state_dict()), (model_type + "/epoch{}.pth".format(epoch)))

    return training_loss_write,training_mse_write, training_ssim_write, training_hybrid_write,testing_loss_write, testing_mse_write, testing_ssim_write, testing_hybrid_write

def test_W_crop_W_TimeVariant(DataLoader, model,model_type, criterion,epoch, epoch_first_stage, PlotOption=False):
    model.eval()
    total_testing_sample = 0.0
    # testing_acc = 0.0
    testing_loss = 0.0
    testing_mse = 0.0
    testing_ssim = 0.0
    testing_hybrid =0.0

    with torch.no_grad():
        # for j, data_test in enumerate(test_loader):

        for i, data_test in enumerate(DataLoader):
            bscan_test, y_true, crop_y = data_test
            bscan_test = bscan_test.float().cuda()
            y_true = y_true.cuda()
            crop_y = crop_y.cuda()

            # calculate outputs by running images through the network
            outputs_test = model(bscan_test)
            hybrid,ssim, mse = criterion(outputs_test, y_true, crop_y)

            if epoch<epoch_first_stage:
                loss2 = mse
            else:
                loss2 = hybrid


            testing_loss += loss2.item() * y_true.size(0)
            testing_mse += mse.item() * y_true.size(0)
            testing_ssim += ssim.item()* y_true.size(0)
            testing_hybrid += hybrid.item()*y_true.size(0)

            total_testing_sample += y_true.size(0)

            if (PlotOption == True) & (i<1) :
                x_val_np = bscan_test.cpu().numpy()
                y_true_np = y_true.cpu().numpy()
                y_pred_np = outputs_test.cpu().numpy()

                for ii_sample in range(len(x_val_np)):
                    output_folder = model_type + '/interation_figure'
                    os.makedirs(output_folder, exist_ok=True)
                    sample_folder = os.path.join(output_folder, f'batch0_sample_{ii_sample}')
                    os.makedirs(sample_folder, exist_ok=True)
                    figure_name = os.path.join(sample_folder, f'epoch{epoch}.png')

                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 3, 1)
                    plt.title('Input Image1')
                    plt.imshow(x_val_np[ii_sample].squeeze(), cmap='gray')

                    # plt.subplot(1, 3, 2)
                    # plt.title('Input Image2')
                    # plt.imshow(input_image[:,:,1], cmap='gray')

                    plt.subplot(1, 3, 2)
                    plt.title('Ground Truth')
                    plt.imshow(y_true_np[ii_sample].squeeze(), cmap='gray')

                    plt.subplot(1, 3, 3)
                    plt.title('Predicted Image')
                    plt.imshow(y_pred_np[ii_sample].squeeze(), cmap='gray')
                    plt.savefig(figure_name)

                    plt.close()

            # if PlotOption == True:
            #     ylabel.extend(testing_groundtruth.cpu().numpy())
            #     ypred.extend(testing_predicted.cpu().numpy())

        testing_loss = testing_loss / total_testing_sample
        testing_mse = testing_mse/ total_testing_sample
        testing_ssim = testing_ssim/total_testing_sample
        testing_hybrid = testing_hybrid/total_testing_sample
    # if PlotOption==True:
    #     return testing_acc, testing_loss, ylabel, ypred
    # else:
    return testing_loss, testing_mse, testing_ssim, testing_hybrid

def check_cuda():
    _cuda = False
    if torch.cuda.is_available():
        _cuda = True
    return _cuda