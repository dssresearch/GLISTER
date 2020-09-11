import copy
import datetime
import numpy as np
import os
import subprocess
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from models.simpleNN_net import * #ThreeLayerNet
from models.logistic_regression import LogisticRegNet
from models.set_function_all import SetFunctionFacLoc, SetFunctionTaylor, SetFunctionTaylorDeep, SetFunctionBatch # as SetFunction #SetFunctionCompare
from models.set_function_ideas import SetFunctionTaylorDeep_ReLoss_Mean
from sklearn.model_selection import train_test_split
from utils.custom_dataset import load_dataset_numpy
from custom_dataset_old import load_dataset_numpy as load_dataset_numpy_old
import math
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# #device = "cpu"
print("Using Device:", device)

## Convert to this argparse 
datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
select_every = int(sys.argv[5])#70
warm_method = 0  # whether to use warmstart-onestep (1) or online (0)
num_runs = 1    # number of random runs

# datadir = sys.argv[1]
# data_name = sys.argv[2]
# fraction = float(sys.argv[3])
# num_epochs = int(sys.argv[4])
# select_every = int(sys.argv[5])
# warm_method = int(sys.argv[6])  # whether to use warmstart-onestep (1) or online (0)
# num_runs = int(sys.argv[7])    # number of random runs
learning_rate = 0.05
all_logs_dir = './results/Check/' + data_name + '/' + str(fraction) + '/' + str(select_every)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt') 
logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) +  \
    '_selEvery:' + str(select_every) + '_variant' + str(warm_method) + '_runs' + str(num_runs)
print(exp_name)
exp_start_time = datetime.datetime.now()
print("=======================================", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)

#print("Datadir",datadir,"Data name",data_name)

if data_name in ['dna','sklearn-digits','satimage','svmguide1','letter','shuttle','ijcnn1','sensorless','connect_4','sensit_seismic']:
    fullset, valset, testset, num_cls = load_dataset_numpy_old(datadir, data_name)
else:
    fullset, valset, testset, num_cls = load_dataset_numpy(datadir, data_name)


if data_name == 'mnist':    
    x_trn, y_trn = fullset.data, fullset.targets
    x_tst, y_tst = testset.data, testset.targets
    x_trn = x_trn.view(x_trn.shape[0], -1)
    x_tst = x_tst.view(x_tst.shape[0], -1)
    # Get validation data: Its 10% of the entire (full) training data
    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
else:
    x_trn, y_trn = torch.from_numpy(fullset[0]).float(), torch.from_numpy(fullset[1]).long()
    x_tst, y_tst = torch.from_numpy(testset[0]).float(), torch.from_numpy(testset[1]).long()
    x_val, y_val = torch.from_numpy(valset[0]).float(), torch.from_numpy(valset[1]).long()
"""
else:
    x_trn, y_trn = fullset
    x_tst, y_tst = testset
    # Load as a Pytorch Tensor
    x_trn = torch.from_numpy(x_trn.astype(np.float32))
    x_tst = torch.from_numpy(x_tst.astype(np.float32))
    y_trn = torch.from_numpy(y_trn.astype(np.int64))
    y_tst = torch.from_numpy(y_tst.astype(np.int64))
    # Get validation data: Its 10% of the entire (full) training data
    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
"""
print('-----------------------------------------')
print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
#print(y_trn.shape, y_val.shape, y_tst.shape)

N, M = x_trn.shape
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)
# Transfer all the data to GPU
d_t = time.time()
x_trn, y_trn = x_trn.to(device), y_trn.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)
print("Transferred data to device in time:", time.time()-d_t)
print_every = 50
train_batch_size_for_greedy = 1200
train_loader_greedy = []
for item in range(math.ceil(len(x_trn) / train_batch_size_for_greedy)):
    inputs = x_trn[item * train_batch_size_for_greedy:(item + 1) * train_batch_size_for_greedy]
    target = y_trn[item * train_batch_size_for_greedy:(item + 1) * train_batch_size_for_greedy]
    train_loader_greedy.append((inputs, target))

valid_loader = []
train_batch_size = 128
for item in range(math.ceil(len(x_val)/train_batch_size)):
    inputs = x_val[item*train_batch_size:(item+1)*train_batch_size]
    target  = y_val[item*train_batch_size:(item+1)*train_batch_size]
    valid_loader.append((inputs,target))

train_loader = []
for item in range(math.ceil(len(x_trn)/train_batch_size)):
    inputs = x_trn[item*train_batch_size:(item+1)*train_batch_size]
    target  = y_trn[item*train_batch_size:(item+1)*train_batch_size]
    train_loader.append((inputs,target))

'''test_loader = []
for item in range(math.ceil(len(x_tst)/test_batch_size)):
    inputs = x_tst[item*test_batch_size:(item+1)*test_batch_size]
    target  = y_tst[item*test_batch_size:(item+1)*test_batch_size]
    test_loader.append((inputs,target))'''

def train_model_step_taylor(bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    model = TwoLayerNet(M, num_cls, 100)
    #model = LogisticRegNet(M, num_cls)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model = model.to(device)

    step_bud = math.ceil(bud*select_every/ num_epochs)
    remainList = set(list(range(N)))

    idxs = list(np.random.choice(N, size=step_bud, replace=False))
    remainList = remainList.difference(idxs)


    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    #setf_model = SetFunctionTaylorDeep(train_loader_greedy, valid_loader, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N)

    setf_model = SetFunctionTaylor(x_trn, y_trn, x_val, y_val, valid, model,
                                   criterion, criterion_nored, learning_rate, device)

    #setf_model = SetFunctionTaylorDeep_ReLoss_Mean(x_trn, y_trn, train_batch_size_for_greedy, x_val, y_val, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N) 

    print("Starting Online Stepwise OneStep Run with taylor!")
    # if valid:
    #     print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    # else:
    #     print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)

    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    # idxs = start_rand_idxs
    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)

        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if ((i + 1) % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            # print("With Taylor approximation",file=logfile)
            # print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs = setf_model.naive_greedy_max(step_bud, clone_dict,list(remainList))  # , grads_idxs
            remainList = remainList.difference(new_idxs)
            idxs.extend(new_idxs) 

            model.load_state_dict(cached_state_dict)

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn).mean()
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100 * val_correct / val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs

def train_model_full_one_step(start_rand_idxs, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    setf_model = SetFunctionBatch(x_trn, y_trn, x_val, y_val, valid, model, 
        criterion, criterion_nored, learning_rate, device)

    print("Starting Online OneStep Run without taylor!")
    # if valid:
    #     print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    # else:
    #     print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)

    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    # idxs = start_rand_idxs
    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)

        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if ((i + 1) % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            # print("With Taylor approximation",file=logfile)
            # print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs = setf_model.naive_greedy_max(bud, clone_dict)  # , grads_idxs
            idxs = new_idxs  # update the current set
            # print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
            # print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
            model.load_state_dict(cached_state_dict)

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn).mean()
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100 * val_correct / val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs

def train_model_online_taylor_reloss(start_rand_idxs, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    setf_model = SetFunctionTaylorDeep_ReLoss_Mean(x_trn, y_trn, train_batch_size_for_greedy, x_val, y_val, valid, model, 
            criterion, criterion_nored, learning_rate, device, N) 

    print("Starting Online OneStep Run with taylor on logit!")
    # if valid:
    #     print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    # else:
    #     print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)

    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    # idxs = start_rand_idxs
    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)

        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if ((i + 1) % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            # print("With Taylor approximation",file=logfile)
            # print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs = setf_model.naive_greedy_max(bud, clone_dict)  # , grads_idxs
            idxs = new_idxs  # update the current set
            # print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
            # print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
            model.load_state_dict(cached_state_dict)

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn).mean()
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100 * val_correct / val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs

def train_model_online_taylor_deep(start_rand_idxs, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    setf_model = SetFunctionTaylorDeep(train_loader_greedy, valid_loader, valid, model, 
            criterion, criterion_nored, learning_rate, device, N)

    #setf_model = SetFunctionTaylor(x_trn, y_trn, x_val, y_val, valid, model,
    #                               criterion, criterion_nored, learning_rate, device)

    #setf_model = SetFunctionTaylorDeep_ReLoss_Mean(x_trn, y_trn, train_batch_size_for_greedy, x_val, y_val, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N) 

    print("Starting Online OneStep Run with taylor on loss with batches!")
    # if valid:
    #     print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    # else:
    #     print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)

    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    # idxs = start_rand_idxs

    train_sub_loader = []
    x_trn_sub = x_trn[idxs]
    y_trn_sub = y_trn[idxs]
    for item in range(math.ceil(len(x_trn_sub)/train_batch_size)):
          inputs = x_trn_sub[item*train_batch_size:(item+1)*train_batch_size]
          target  = y_trn_sub[item*train_batch_size:(item+1)*train_batch_size]
          train_sub_loader.append((inputs,target))

    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        total_subset_loss  = 0
        for idx, data in  enumerate(train_sub_loader, 0):    
            inputs, target = data
            #inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            scores = model(inputs)
            loss = criterion(scores, target)
            total_subset_loss += loss.item()
            loss.backward()
            optimizer.step()

        substrn_losses[i] = (1.0*total_subset_loss)/(idx+1)

        with torch.no_grad():
            total_train_loss = 0
            for idx, train_data in  enumerate(train_loader, 0):
                inputs, target = train_data
                #inputs, target = inputs.to(device), target.to(device)
                scores = model(inputs)
                tr_loss = criterion(scores, target)
                total_train_loss += tr_loss.item()

            full_trn_loss = (1.0*total_train_loss)/(idx+1)
            
            total_valid_loss = 0
            for idx, val_data in  enumerate(valid_loader, 0):
                inputs, target = val_data
                #inputs, target = inputs.to(device), target.to(device)
                scores = model(inputs)
                val_loss = criterion(scores, target)
                total_valid_loss += val_loss.item()
            
            val_loss = (1.0*total_valid_loss)/(idx+1)
            
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i, 'SubsetTrn,FullTrn,ValLoss:', substrn_losses[i], full_trn_loss, val_loss)

        if ((i + 1) % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            # print("With Taylor approximation",file=logfile)
            # print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs = setf_model.naive_greedy_max(bud, clone_dict)  # , grads_idxs
            idxs =  new_idxs # update the current set
            random.shuffle(idxs)
            # print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
            # print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
            model.load_state_dict(cached_state_dict)

            train_sub_loader = []
            x_trn_sub = x_trn[idxs]
            y_trn_sub = y_trn[idxs]
            for item in range(math.ceil(len(x_trn_sub)/train_batch_size)):
                  inputs = x_trn_sub[item*train_batch_size:(item+1)*train_batch_size]
                  target  = y_trn_sub[item*train_batch_size:(item+1)*train_batch_size]
                  train_sub_loader.append((inputs,target))

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn).mean()
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100 * val_correct / val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs

def train_model_online_taylor(start_rand_idxs, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    #setf_model = SetFunctionTaylorDeep(train_loader_greedy, valid_loader, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N)

    setf_model = SetFunctionTaylor(x_trn, y_trn, x_val, y_val, valid, model,
                                   criterion, criterion_nored, learning_rate, device)

    #setf_model = SetFunctionTaylorDeep_ReLoss_Mean(x_trn, y_trn, train_batch_size_for_greedy, x_val, y_val, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N) 

    print("Starting Online OneStep Run with taylor on loss!")
    # if valid:
    #     print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    # else:
    #     print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)

    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    # idxs = start_rand_idxs
    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)

        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if ((i + 1) % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            # print("With Taylor approximation",file=logfile)
            # print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs = setf_model.naive_greedy_max(bud, clone_dict)  # , grads_idxs
            idxs = new_idxs  # update the current set
            # print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
            # print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
            model.load_state_dict(cached_state_dict)

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn).mean()
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100 * val_correct / val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs


def random_greedy_train_model_online_taylor(start_rand_idxs, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model = model.to(device)
    idxs = start_rand_idxs
    total_idxs = list(np.arange(len(y_trn)))
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    #setf_model = SetFunctionTaylorDeep(train_loader_greedy, valid_loader, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N)
    
    setf_model = SetFunctionTaylor(x_trn, y_trn, x_val, y_val, valid, model,
                                   criterion, criterion_nored, learning_rate, device)

    #setf_model = SetFunctionTaylorDeep_ReLoss_Mean(x_trn, y_trn, train_batch_size_for_greedy, x_val, y_val, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N) 
    
    print("Starting Randomized Greedy Online OneStep Run with taylor!")
    # if valid:
    #     print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    # else:
    #     print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)

        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if ((i + 1) % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            # print("With Taylor approximation",file=logfile)
            # print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs = setf_model.naive_greedy_max(int(0.9 * bud), clone_dict)
            rem_idxs = list(set(total_idxs).difference(set(new_idxs)))
            new_idxs.extend(list(np.random.choice(rem_idxs, size=int(0.1 * bud), replace=False)))
            # , grads_idxs
            idxs = new_idxs  # update the current set
            # print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
            # print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
            model.load_state_dict(cached_state_dict)

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn).mean()
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100 * val_correct / val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs


def facloc_reg_train_model_online_taylor(start_rand_idxs, fac_loc_idx, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
       print("Using:", torch.cuda.device_count(), "GPUs!")
       model = nn.DataParallel(model)
       cudnn.benchmark = True
    x_val1 = torch.cat([x_val, x_trn[fac_loc_idx]], dim=0)
    y_val1 = torch.cat([y_val, y_trn[fac_loc_idx]], dim=0)
    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    #setf_model = SetFunctionTaylorDeep(train_loader_greedy, valid_loader, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N)

    setf_model = SetFunctionTaylor(x_trn, y_trn, x_val1, y_val1, valid, model,
        criterion, criterion_nored, learning_rate, device)

    #setf_model = SetFunctionTaylorDeep_ReLoss_Mean(x_trn, y_trn, train_batch_size_for_greedy, x_val, y_val, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N) 


    print("Starting Facility Location Regularized Online OneStep Run with taylor!")
    # if valid:
    #     print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    # else:
    #     print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)
    
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    # idxs = start_rand_idxs
    for i in range(num_epochs):    
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)    
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)

        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i+1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if ((i+1) % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            #print("With Taylor approximation",file=logfile)
            #print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs = setf_model.naive_greedy_max(bud, clone_dict) #, grads_idxs
            idxs = new_idxs     # update the current set
            #print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
            # print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
            model.load_state_dict(cached_state_dict)
    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn).mean()
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100*val_correct/val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)   
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct/total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs


def train_model_mod_taylor(start_rand_idxs, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    #setf_model = SetFunctionTaylorDeep(train_loader_greedy, valid_loader, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N)

    setf_model = SetFunctionTaylor(x_trn, y_trn, x_val, y_val, valid, model,
                                   criterion, criterion_nored, learning_rate, device)

    #setf_model = SetFunctionTaylorDeep_ReLoss_Mean(x_trn, y_trn, train_batch_size_for_greedy, x_val, y_val, valid, model, 
    #        criterion, criterion_nored, learning_rate, device, N) 

    print("Starting Online OneStep Run after optimization with taylor!")
    # if valid:
    #     print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    # else:
    #     print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)

    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    # idxs = start_rand_idxs

    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn, y_trn
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

    for i in range(int(num_epochs/20)):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)

        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if ((i + 1) % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            # print("With Taylor approximation",file=logfile)
            # print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs = setf_model.naive_greedy_max(bud, clone_dict)  # , grads_idxs
            idxs = new_idxs  # update the current set
            # print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
            # print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
            model.load_state_dict(cached_state_dict)

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn).mean()
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100 * val_correct / val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs



def train_model_online_random(start_rand_idxs, bud):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
       print("Using:", torch.cuda.device_count(), "GPUs!")
       model = nn.DataParallel(model)
       cudnn.benchmark = True
    
    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("Starting Random Selection with taylor!")
    
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    for i in range(num_epochs):    
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)    
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)

        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i+1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if ((i+1) % select_every == 0):
            state = np.random.get_state()
            np.random.seed(i)
            idxs = np.random.choice(N, size=bud, replace=False)
            np.random.set_state(state)


    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn).mean()
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100*val_correct/val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)   
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct/total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses


def train_model_random(start_rand_idxs):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
       print("Using:", torch.cuda.device_count(), "GPUs!")
       model = nn.DataParallel(model)
       cudnn.benchmark = True
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # idxs = np.random.randint(N, size=bud)
    idxs = start_rand_idxs
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    
    print("Starting Random Run!")
    for i in range(num_epochs):    
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets) 
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)
        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i+1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100*val_correct/val_total
    
    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets) 
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct/total

    print("RandRun---------------------------------",)
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc )
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses


def train_model_Facloc():
    torch.manual_seed(42)
    np.random.seed(42)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # idxs = np.random.randint(N, size=bud)

    setf_model = SetFunctionFacLoc(device, train_loader_greedy)
    if fraction != 1:
        idxs = setf_model.lazy_greedy_max(bud, model)
    print("Starting Facility Location!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)

    print("Starting Random Run!")
    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)
        substrn_losses[i] = loss.item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        full_trn_out = model(x_trn)
        full_trn_loss = criterion(full_trn_out, y_trn)
        sub_trn_out = model(x_trn[idxs])
        sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
        val_out = model(x_val)
        val_loss = criterion(val_out, y_val)
        _, val_predict = val_out.max(1)
        val_correct = val_predict.eq(y_val).sum().item()
        val_total = y_val.size(0)
    val_acc = 100 * val_correct / val_total

    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = x_tst.to(device), y_tst.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    tst_acc = 100.0 * correct / total

    print("RandRun---------------------------------", )
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs



start_idxs = np.random.choice(N, size=bud, replace=False)
random_subset_idx = [x for x in start_idxs]
# Random Run
rv1, rt1, rv2, rt2, rand_substrn_losses, rand_fulltrn_losses, rand_val_losses = train_model_random(start_idxs)
# Facility Location Run
fv1, ft1, fv2, ft2, facloc_substrn_losses, facloc_fulltrn_losses, facloc_val_losses, facloc_idxs = train_model_Facloc()
# Online algo run
#ft_val_valacc, ft_val_tstacc, ft_val_valloss, ft_val_tstloss, ftay_fval_substrn_losses, ftay_fval_fulltrn_losses, ftay_fval_val_losses, fsubset_idxs = train_model_full_one_step(start_idxs, bud, True)
# Online algo run
#rt_val_valacc, rt_val_tstacc, rt_val_valloss, rt_val_tstloss, rtay_fval_substrn_losses, rtay_fval_fulltrn_losses, rtay_fval_val_losses, rsubset_idxs = train_model_online_taylor_reloss(start_idxs, bud, True)
# Online algo run
t_val_valacc, t_val_tstacc, t_val_valloss, t_val_tstloss, tay_fval_substrn_losses, tay_fval_fulltrn_losses, tay_fval_val_losses, subset_idxs = train_model_online_taylor(start_idxs, bud, True)
# Online algo run
dt_val_valacc, dt_val_tstacc, dt_val_valloss, dt_val_tstloss, dtay_fval_substrn_losses, dtay_fval_fulltrn_losses, dtay_fval_val_losses, dsubset_idxs = train_model_online_taylor_deep(start_idxs, bud, True)
# Stepwise algo run
step_t_val_valacc, step_t_val_tstacc, step_t_val_valloss, step_t_val_tstloss, step_tay_fval_substrn_losses, step_tay_fval_fulltrn_losses, step_tay_fval_val_losses, step_subset_idxs = train_model_step_taylor(bud, True)
#Facility Location OneStep Runs
facloc_reg_t_val_valacc, facloc_reg_t_val_tstacc, facloc_reg_t_val_valloss, facloc_reg_t_val_tstloss, facloc_reg_tay_fval_substrn_losses, facloc_reg_tay_fval_fulltrn_losses, facloc_reg_tay_fval_val_losses, facloc_reg_subset_idxs = facloc_reg_train_model_online_taylor(start_idxs, facloc_idxs, bud, True)
#Randomized Greedy Taylor OneStep Runs
rand_t_val_valacc, rand_t_val_tstacc, rand_t_val_valloss, rand_t_val_tstloss, rand_tay_fval_substrn_losses, rand_tay_fval_fulltrn_losses, rand_tay_fval_val_losses, rand_subset_idxs = random_greedy_train_model_online_taylor(start_idxs, bud, True)
#Modified OneStep Runs
#mod_t_val_valacc, mod_t_val_tstacc, mod_t_val_valloss, mod_t_val_tstloss, mod_tay_fval_substrn_losses, mod_tay_fval_fulltrn_losses, mod_tay_fval_val_losses, mod_subset_idxs = train_model_mod_taylor(start_idxs, bud, True)


plot_start_epoch = 0
###### Subset Trn loss with val = VAL #############
plt.figure()
# plt.plot(np.arange(1,num_epochs+1), knn_fval_substrn_losses, 'r-', label='knn_v=val')
plt.plot(np.arange(plot_start_epoch,num_epochs), tay_fval_substrn_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch,num_epochs), step_tay_fval_substrn_losses[plot_start_epoch:], 'r-', label='step_tay_v=val')
plt.plot(np.arange(plot_start_epoch,num_epochs), rand_substrn_losses[plot_start_epoch:], 'g-', label='random')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_substrn_losses[plot_start_epoch:], 'pink', label='FacLoc')
#plt.plot(np.arange(plot_start_epoch,num_epochs), mod_tay_fval_substrn_losses[plot_start_epoch:], 'c', label='after_opt_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_tay_fval_substrn_losses[plot_start_epoch:], 'k-', label='rand_reg_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_tay_fval_substrn_losses[plot_start_epoch:], 'y', label='facloc_reg_tay_v=val')
# plt.plot(np.arange(1,num_epochs+1), fc_substrn_losses, '#000000', label='Facility Location')
#plt.plot(np.arange(plot_start_epoch,num_epochs), ftay_fval_substrn_losses[plot_start_epoch:], 'orange', label='NON-tay_v=val')
#plt.plot(np.arange(plot_start_epoch,num_epochs), rtay_fval_substrn_losses[plot_start_epoch:], 'm-', label='logit_tay_v=val')
plt.plot(np.arange(plot_start_epoch,num_epochs), dtay_fval_substrn_losses[plot_start_epoch:], '#8c564b', label='deep_tay_v=val')


plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Subset trn loss')
plt.title('Subset Training Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'substrn_loss_v=VAL.png'
plt.savefig(plt_file)
plt.clf()


########################################################################
###### Full Trn loss with val = VAL #############
plt.figure()
# plt.plot(np.arange(1,num_epochs+1), knn_fval_fulltrn_losses, 'r-', label='knn_v=val')

plt.plot(np.arange(plot_start_epoch, num_epochs), tay_fval_fulltrn_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), step_tay_fval_fulltrn_losses[plot_start_epoch:], 'r-', label='step_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_fulltrn_losses[plot_start_epoch:], 'g-', label='random')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_fulltrn_losses[plot_start_epoch:], 'pink', label='FacLoc')
#plt.plot(np.arange(plot_start_epoch, num_epochs), mod_tay_fval_fulltrn_losses[plot_start_epoch:], 'c', label='after_opt_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_tay_fval_fulltrn_losses[plot_start_epoch:], 'k-', label='rand_reg_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_tay_fval_fulltrn_losses[plot_start_epoch:], 'y', label='facloc_reg_tay_v=val')
#plt.plot(np.arange(plot_start_epoch, num_epochs), ftay_fval_fulltrn_losses[plot_start_epoch:], 'orange', label='NON-tay_v=val')
#plt.plot(np.arange(plot_start_epoch, num_epochs), rtay_fval_fulltrn_losses[plot_start_epoch:], 'm-', label='logit_tay_v=val')
plt.plot(np.arange(plot_start_epoch,num_epochs), dtay_fval_fulltrn_losses[plot_start_epoch:], '#8c564b', label='deep_tay_v=val')


plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Full trn loss')
plt.title('Full Training Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'fulltrn_loss_v=VAL.png'
plt.savefig(plt_file)
plt.clf()


########################################################################
###### Validation loss with val = VAL #############
plt.figure()
# plt.plot(np.arange(1,num_epochs+1), knn_fval_val_losses, 'r-', label='knn_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), tay_fval_val_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), step_tay_fval_val_losses[plot_start_epoch:], 'r-', label='step_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_val_losses[plot_start_epoch:], 'g-', label='random')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_val_losses[plot_start_epoch:], 'pink', label='FacLoc')
#plt.plot(np.arange(plot_start_epoch,num_epochs), mod_tay_fval_val_losses[plot_start_epoch:], 'c', label='after_opt_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_tay_fval_val_losses[plot_start_epoch:], 'k-', label='rand_reg_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_tay_fval_val_losses[plot_start_epoch:], 'y', label='facloc_reg_tay_v=val')
#plt.plot(np.arange(plot_start_epoch, num_epochs), ftay_fval_val_losses[plot_start_epoch:], 'orange', label='NON-tay_v=val')
#plt.plot(np.arange(plot_start_epoch, num_epochs), rtay_fval_val_losses[plot_start_epoch:], 'm-', label='logit_tay_v=val')
plt.plot(np.arange(plot_start_epoch,num_epochs), dtay_fval_val_losses[plot_start_epoch:], '#8c564b', label='deep_tay_v=val')


plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.title('Validation Loss vs Epochs ' + data_name + '_' + str(fraction)+ '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'valloss_v=VAL.png'
plt.savefig(plt_file)
plt.clf()

print(data_name,":Budget = ", fraction, file=logfile)
print('---------------------------------------------------------------------',file=logfile)
print('|Algo                            | Val Acc       |   Test Acc       |',file=logfile)
print('| -------------------------------|:-------------:| ----------------:|',file=logfile)
print('*| Facility Location             |',fv1, '  | ',ft1,' |',file=logfile)
print('*| Taylor with Validation=VAL     |', t_val_valacc , '  | ', t_val_tstacc ,' |',file=logfile)
#print('*| Without Taylor with Validation=VAL     |', ft_val_valacc , '  | ', ft_val_tstacc ,' |',file=logfile)
#print('*| Taylor on logit with Validation=VAL     |', rt_val_valacc , '  | ', rt_val_tstacc ,' |',file=logfile)
print('*|  Batched Taylor with Validation=VAL     |', dt_val_valacc , '  | ', dt_val_tstacc ,' |',file=logfile)
print('*| Stepwise Taylor with Validation=VAL     |', step_t_val_valacc , '  | ', step_t_val_tstacc ,' |',file=logfile)
print('*| Random Selection               |', rv1,              '  | ', rt1,              ' |',file=logfile)
#print('*| Taylor after training               |', mod_t_val_valacc,'  | ', mod_t_val_tstacc,' |',file=logfile)
print('*| random regularized Taylor after training               |', rand_t_val_valacc,'  | ', rand_t_val_tstacc,' |',file=logfile)
print('*| facloc regularizec Taylor after training               |', facloc_reg_t_val_valacc,'  | ', facloc_reg_t_val_tstacc,' |',file=logfile)
print("\n", file=logfile)

print("=========Random Results==============", file=logfile)
print("*Rand Validation LOSS:", rv2, file=logfile)
print("*Rand Test Data LOSS:", rt2, file=logfile)
print("*Rand Full Trn Data LOSS:", rand_fulltrn_losses[-1], file=logfile)

print("=========FacLoc Results==============", file=logfile)
print("*Facloc Validation LOSS:", fv2, file=logfile)
print("*Facloc Test Data LOSS:", ft2, file=logfile)
print("*Facloc Full Trn Data LOSS:", facloc_fulltrn_losses[-1], file=logfile)

print("=========Online Selection Taylor with Validation Set===================", file=logfile)      
print("*Taylor v=VAL Validation LOSS:", t_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", t_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", tay_fval_fulltrn_losses[-1], file=logfile)

'''print("=========Online Selection Without Taylor with Validation Set===================", file=logfile)      
print("*Taylor v=VAL Validation LOSS:", ft_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", ft_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", ftay_fval_fulltrn_losses[-1], file=logfile)

print("=========Online Selection Taylor on logit with Validation Set===================", file=logfile)      
print("*Taylor v=VAL Validation LOSS:", rt_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", rt_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", rtay_fval_fulltrn_losses[-1], file=logfile)'''

print("=========Batched Taylor on logit with Validation Set===================", file=logfile)      
print("*Taylor v=VAL Validation LOSS:", dt_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", dt_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", dtay_fval_fulltrn_losses[-1], file=logfile)

print("=========Stepwise Online Selection Taylor with Validation Set===================", file=logfile)      
print("*Taylor v=VAL Validation LOSS:", step_t_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", step_t_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", step_tay_fval_fulltrn_losses[-1], file=logfile)

print("=========Random Regularized Online Selection Taylor with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", rand_t_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", rand_t_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", rand_tay_fval_fulltrn_losses[-1], file=logfile)

print("=========Facility Location Loss regularized Online Selection Taylor with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", facloc_reg_t_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", facloc_reg_t_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", facloc_reg_tay_fval_fulltrn_losses[-1], file=logfile)

'''print("=========Online Selection Taylor after model training with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", mod_t_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", mod_t_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", mod_tay_fval_fulltrn_losses[-1], file=logfile)'''
print("=============================================================================================", file=logfile) 
print("---------------------------------------------------------------------------------------------", file=logfile)
print("\n", file=logfile)       

'''mod_subset_idxs = list(mod_subset_idxs)
#print(len(mod_subset_idxs))
with open(all_logs_dir+'/mod_one_step_subset_selected.txt', 'w') as log_file:
    print(mod_subset_idxs, file=log_file)'''

subset_idxs = list(subset_idxs)
with open(all_logs_dir+'/one_step_subset_selected.txt', 'w') as log_file:
    print(subset_idxs, file=log_file)

'''fsubset_idxs = list(fsubset_idxs)
with open(all_logs_dir+'/without_taylor_subset_selected.txt', 'w') as log_file:
    print(fsubset_idxs, file=log_file)

rsubset_idxs = list(rsubset_idxs)
with open(all_logs_dir+'/taylor_logit_subset_selected.txt', 'w') as log_file:
    print(rsubset_idxs, file=log_file)'''

dsubset_idxs = list(dsubset_idxs)
with open(all_logs_dir+'/taylor_logit_subset_selected.txt', 'w') as log_file:
    print(dsubset_idxs, file=log_file)

step_subset_idxs = list(step_subset_idxs)
#print(len(step_subset_idxs))
with open(all_logs_dir+'/stepwise_one_step_subset_selected.txt', 'w') as log_file:
    print(step_subset_idxs, file=log_file)

rand_subset_idxs = list(rand_subset_idxs)
with open(all_logs_dir+'/rand_reg_one_step_subset_selected.txt', 'w') as log_file:
    print(rand_subset_idxs, file=log_file)

facloc_reg_subset_idxs = list(facloc_reg_subset_idxs)
with open(all_logs_dir+'/facloc_reg_one_step_subset_selected.txt', 'w') as log_file:
    print(facloc_reg_subset_idxs, file=log_file)

random_subset_idx = list(random_subset_idx)
with open(all_logs_dir+'/random_subset_selected.txt', 'w') as log_file:
    print(random_subset_idx, file=log_file)

facloc_idxs = list(facloc_idxs)
with open(all_logs_dir+'/facloc_subset_selected.txt', 'w') as log_file:
    print(facloc_idxs, file=log_file)
