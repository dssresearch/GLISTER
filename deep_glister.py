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
from torch.utils.data.sampler import SubsetRandomSampler
from models.set_function_grad_computation_taylor import GlisterSetFunction as SetFunction, Glister_Linear_SetFunction_Closed as ClosedSetFunction
from models.mnist_net import MnistNet
from models.resnet import ResNet18
from utils.custom_dataset import load_mnist_cifar
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler


def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss


def write_knndata(datadir, x_trn, y_trn, x_val, y_val, x_tst, y_tst, dset_name):
    ## Create VAL data
    subprocess.run(["mkdir", "-p", datadir])
    # x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
    trndata = np.c_[x_trn.cpu().numpy(), y_trn.cpu().numpy()]
    valdata = np.c_[x_val.cpu().numpy(), y_val.cpu().numpy()]
    tstdata = np.c_[x_tst.cpu().numpy(), y_tst.cpu().numpy()]
    # Write out the trndata
    trn_filepath = os.path.join(datadir, 'knn_' + dset_name + '.trn')
    val_filepath = os.path.join(datadir, 'knn_' + dset_name + '.val')
    tst_filepath = os.path.join(datadir, 'knn_' + dset_name + '.tst')
    np.savetxt(trn_filepath, trndata, fmt='%.6f')
    np.savetxt(val_filepath, valdata, fmt='%.6f')
    np.savetxt(tst_filepath, tstdata, fmt='%.6f')
    return


def perform_knnsb_selection(datadir, dset_name, budget, selUsing):
    trn_filepath = os.path.join(datadir, 'knn_' + dset_name + '.trn')
    if selUsing == 'val':
        val_filepath = os.path.join(datadir, 'knn_' + dset_name + '.val')
    else:
        val_filepath = trn_filepath

    run_path = './run_data/'
    output_dir = run_path + 'KNNSubmod_' + dset_name + '/'
    indices_file = output_dir + 'KNNSubmod_' + str((int)(budget*100)) + ".subset"
    subprocess.call(["mkdir", output_dir])
    knnsb_args = []
    knnsb_args.append('../build/KNNSubmod')
    knnsb_args.append(trn_filepath)
    knnsb_args.append(val_filepath)
    knnsb_args.append(" ")  # File delimiter!!
    knnsb_args.append(str(budget))
    knnsb_args.append(indices_file)
    knnsb_args.append("1")  # indicates cts data. Deprecated.
    print("Obtaining the subset")
    subprocess.run(knnsb_args)
    print("finished selection")
    # Can make it return the indices_file if using with other function.
    idxs_knnsb = np.genfromtxt(indices_file, delimiter=',', dtype=int) # since they are indices!
    return idxs_knnsb

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:0"
print("Using Device:", device)

## Convert to this argparse
datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
select_every = int(sys.argv[5])
feature = sys.argv[6]# 70
warm_method = 0  # whether to use warmstart-onestep (1) or online (0)
num_runs = 1  # number of random runs
learning_rate = 0.05
all_logs_dir = './results/debugging/' + data_name +'_grad/' + feature +'/' + str(fraction) + '/' + str(select_every)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
           '_selEvery:' + str(select_every) + '_variant' + str(warm_method) + '_runs' + str(num_runs)
print(exp_name)
exp_start_time = datetime.datetime.now()
print("=======================================", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)

fullset, valset, testset, num_cls = load_mnist_cifar(datadir, data_name, feature)
# Validation Data set is 10% of the Entire Trainset.
validation_set_fraction = 0.1
num_fulltrn = len(fullset)
num_val = int(num_fulltrn * validation_set_fraction)
num_trn = num_fulltrn - num_val
trainset, validset = random_split(fullset, [num_trn, num_val])
trn_batch_size = 20
val_batch_size = 1000
tst_batch_size = 1000

trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                          shuffle=False, pin_memory=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, shuffle=False,
                                               sampler=SubsetRandomSampler(validset.indices),
                                               pin_memory=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                         shuffle=False, pin_memory=True)

trainset_idxs = np.array(trainset.indices)
batch_wise_indices = trainset_idxs[list(BatchSampler(SequentialSampler(trainset_idxs), 1000, drop_last=False))]
cnt = 0
for batch_idx in batch_wise_indices:
    inputs = torch.cat([fullset[x][0].view(1, -1) for x in batch_idx],
                       dim=0).type(torch.float)
    targets = torch.tensor([fullset[x][1] for x in batch_idx])
    if cnt == 0:
        x_trn = inputs
        y_trn = targets
        cnt = cnt + 1
    else:
        x_trn = torch.cat([x_trn, inputs], dim=0)
        y_trn = torch.cat([y_trn, targets], dim=0)
        cnt = cnt + 1

for batch_idx, (inputs, targets) in enumerate(valloader):
    if batch_idx == 0:
        x_val = inputs
        y_val = targets
        x_val_new = inputs.view(val_batch_size, -1)
    else:
        x_val = torch.cat([x_val, inputs], dim=0)
        y_val = torch.cat([y_val, targets], dim=0)
        x_val_new = torch.cat([x_val_new, inputs.view(val_batch_size, -1)], dim=0)
for batch_idx, (inputs, targets) in enumerate(testloader):
    if batch_idx == 0:
        x_tst = inputs
        y_tst = targets
        x_tst_new = inputs.view(tst_batch_size, -1)
    else:
        x_tst = torch.cat([x_tst, inputs], dim=0)
        y_tst = torch.cat([y_tst, targets], dim=0)
        x_tst_new = torch.cat([x_tst_new, inputs.view(tst_batch_size, -1)], dim=0)

write_knndata(datadir, x_trn, y_trn, x_val_new, y_val, x_tst_new, y_tst, data_name)
print('-----------------------------------------')
print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
# print(y_trn.shape, y_val.shape, y_tst.shape)

N, M = x_trn.shape
n_val = x_val_new.shape[0]
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)
# Transfer all the data to GPU
d_t = time.time()
x_trn, y_trn = x_trn.to('cpu'), y_trn.to('cpu')
x_val, y_val = x_val.to('cpu'), y_val.to('cpu')
print("Transferred data to device in time:", time.time() - d_t)
print_every = 3


def train_model_glister(start_rand_idxs, bud):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    if data_name == 'mnist':
        model = MnistNet()
    elif data_name == 'cifar10':
        model = ResNet18(num_cls)
    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    if data_name == 'mnist':
        setf_model = SetFunction(trainset, x_val, y_val, model, criterion,
                             criterion_nored, learning_rate, device, 1, num_cls, 1000)
        num_channels = 1
    elif data_name == 'cifar10':
        setf_model = SetFunction(trainset, x_val, y_val, model, criterion,
                                 criterion_nored, learning_rate, device, 3, num_cls, 1000)
        num_channels = 3
    print("Starting Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        actual_idxs = np.array(trainset.indices)[idxs]
        batch_wise_indices = [actual_idxs[x] for x in list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))]
        subtrn_loss = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [fullset[x][0].view(-1, num_channels, fullset[x][0].shape[1], fullset[x][0].shape[2]) for x in batch_idx],
                dim=0).type(torch.float)
            targets = torch.tensor([fullset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True) # targets can have non_blocking=True.
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()
        val_loss = 0
        full_trn_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)
        if ((i + 1) % select_every) == 0:
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            prev_idxs = idxs
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(bud), clone_dict)
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
            print(len(list(set(prev_idxs).difference(set(idxs)))) + len(list(set(idxs).difference(set(prev_idxs)))))
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total
    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


def train_model_glister_closed(start_rand_idxs, bud):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    if data_name == 'mnist':
        model = MnistNet()
    elif data_name == 'cifar10':
        model = ResNet18(num_cls)
    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    if data_name == 'mnist':
        setf_model = ClosedSetFunction(trainset, x_val, y_val, model, criterion,
                             criterion_nored, learning_rate, device, 1, num_cls, 1000)
        num_channels = 1
    elif data_name == 'cifar10':
        setf_model = ClosedSetFunction(trainset, x_val, y_val, model, criterion,
                                 criterion_nored, learning_rate, device, 3, num_cls, 1000)
        num_channels = 3
    print("Starting Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        actual_idxs = np.array(trainset.indices)[idxs]
        batch_wise_indices = [actual_idxs[x] for x in list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))]
        subtrn_loss = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [fullset[x][0].view(-1, num_channels, fullset[x][0].shape[1], fullset[x][0].shape[2]) for x in batch_idx],
                dim=0).type(torch.float)
            targets = torch.tensor([fullset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True) # targets can have non_blocking=True.
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()
        val_loss = 0
        full_trn_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)
        if ((i + 1) % select_every) == 0:
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            prev_idxs = idxs
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(bud), clone_dict)
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
            print(len(list(set(prev_idxs).difference(set(idxs)))) + len(list(set(idxs).difference(set(prev_idxs)))))
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total
    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


def train_model_mod_online(start_rand_idxs, bud):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    if data_name == 'mnist':
        model = MnistNet()
    elif data_name == 'cifar10':
        model = ResNet18(num_cls)
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting Modified Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #print(batch_idx)
            # targets can have non_blocking=True.
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total

    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


def train_model_random_online(start_rand_idxs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    if data_name == 'mnist':
        model = MnistNet()
    elif data_name == 'cifar10':
        model = ResNet18(num_cls)
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting Random Run")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        subtrn_loss = 0
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            # targets can have non_blocking=True.
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss = 0
        full_trn_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)

        if ((i + 1) % select_every) == 0:
            idxs = np.random.choice(N, size=bud, replace=False)
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs),
                                                           pin_memory=True)

    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end) / 1000

    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total

    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, time


start_idxs = np.random.choice(N, size=bud, replace=False)
random_subset_idx = [trainset.indices[x] for x in start_idxs]

# Online algo run
closed_val_valacc, closed_val_tstacc, closed_val_subtrn_acc, closed_val_full_trn_acc, closed_val_valloss, \
closed_val_tstloss,  closed_val_subtrnloss, closed_val_full_trn_loss, closed_fval_val_losses, \
closed_fval_substrn_losses, closed_fval_fulltrn_losses, closed_subset_idxs, closed_step_time= \
train_model_glister_closed(start_idxs, bud)
#train_model_mod_online(start_idxs, bud)\

print("Closed Form Glister run time: ", closed_step_time)
closed_subset_idxs = [trainset.indices[x] for x in closed_subset_idxs]

#one_val_valacc, one_val_tstacc, one_val_subtrn_acc, one_val_full_trn_acc, one_val_valloss, \
#one_val_tstloss,  one_val_subtrnloss, one_val_full_trn_loss, one_fval_val_losses, \
#one_fval_substrn_losses, one_fval_fulltrn_losses, one_subset_idxs, one_step_time= \
#    train_model_glister(start_idxs, bud)
#print("One step run time: ", one_step_time)
#one_subset_idxs = [trainset.indices[x] for x in one_subset_idxs]

"""
# Modified OneStep Runs
mod_val_valacc, mod_val_tstacc, mod_val_subtrn_acc, mod_val_full_trn_acc, mod_val_valloss, mod_val_tstloss, \
mod_val_subtrnloss, mod_val_full_trn_loss, mod_val_val_losses, mod_val_substrn_losses, mod_val_fulltrn_losses,\
mod_subset_idxs, mod_one_step_time = train_model_mod_online(start_idxs, bud)
print("Mod One Step run time: ", mod_one_step_time)

#Online Random Run
ol_rand_valacc, ol_rand_tstacc, ol_rand_subtrn_acc, ol_rand_full_trn_acc, ol_rand_valloss, ol_rand_tstloss, ol_rand_subtrnloss,\
ol_rand_full_trn_loss, ol_rand_val_losses, ol_rand_substrn_losses, ol_rand_fulltrn_losses, \
ol_random_run_time = train_model_random_online(start_idxs)
print("Online Random Run Time: ", ol_random_run_time)
"""
plot_start_epoch = 0
###### Subset Trn loss with val = VAL #############
plt.figure()
#plt.plot(np.arange(plot_start_epoch, num_epochs), one_fval_substrn_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), closed_fval_substrn_losses[plot_start_epoch:], 'g+', label='closed form taylor')
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
#plt.plot(np.arange(plot_start_epoch, num_epochs), one_fval_fulltrn_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), closed_fval_fulltrn_losses[plot_start_epoch:], 'g+', label='closed form taylor')
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
#plt.plot(np.arange(plot_start_epoch, num_epochs), one_fval_val_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), closed_fval_val_losses[plot_start_epoch:], 'g+', label='closed form taylor')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.title('Validation Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'valloss_v=VAL.png'
plt.savefig(plt_file)
plt.clf()

print(data_name, ":Budget = ", fraction, file=logfile)
print('---------------------------------------------------------------------', file=logfile)
print('|Algo                            | Val Acc       |   Test Acc       |', file=logfile)
print('| -------------------------------|:-------------:| ----------------:|', file=logfile)
#print('*| Taylor with Validation=VAL     |', one_val_valacc, '  | ', one_val_tstacc, ' |', file=logfile)
print('*| Closed form Taylor               |', closed_val_valacc, '  | ', closed_val_tstacc, ' |', file=logfile)

print('---------------------------------------------------', file=logfile)
print('|Algo                            | Run Time       |', file=logfile)
print('| -------------------------------|:-------------:|', file=logfile)
#print('*| Taylor with Validation=VAL     |', one_step_time, '  | ', file=logfile)
print('*| Closed form Taylor               |', closed_step_time, '  | ',file=logfile)
print("\n", file=logfile)

#print("=========Online Selection Taylor with Validation Set===================", file=logfile)
#print("*Taylor v=VAL Validation LOSS:", one_val_valloss, file=logfile)
#print("*Taylor v=VAL Test Data LOSS:", one_val_tstloss, file=logfile)
#print("*Taylor v=VAL Full Trn Data LOSS:", one_fval_fulltrn_losses[-1], file=logfile)

print("=========Closed Form Taylor with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", closed_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", closed_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", closed_fval_fulltrn_losses[-1], file=logfile)

print("=============================================================================================", file=logfile)
print("---------------------------------------------------------------------------------------------", file=logfile)
print("\n", file=logfile)

#subset_idxs = list(one_subset_idxs)
#with open(all_logs_dir + '/one_step_subset_selected.txt', 'w') as log_file:
#    print(subset_idxs, file=log_file)

subset_idxs = list(closed_subset_idxs)
with open(all_logs_dir + '/closed_form_subset_selected.txt', 'w') as log_file:
    print(subset_idxs, file=log_file)
