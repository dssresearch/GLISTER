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
from models.set_function_all import SetFunctionFacLoc
from torch.utils.data.sampler import SubsetRandomSampler
# from models.simpleNN_net import ThreeLayerNet
from models.set_function_stochastic_onestep_taylor import SetFunctionLoader_2 as SetFunction
from models.set_function_stochastic_onestep_taylor import WeightedSetFunctionLoader as WtSetFunction
import math
from models.mnist_net import MnistNet
from utils.data_utils import load_dataset_pytorch
from torch.utils.data import random_split

def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print("Using Device:", device)

## Convert to this argparse
datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
select_every = int(sys.argv[5])  # 70
warm_method = 0  # whether to use warmstart-onestep (1) or online (0)
num_runs = 1  # number of random runs

# datadir = sys.argv[1]
# data_name = sys.argv[2]
# fraction = float(sys.argv[3])
# num_epochs = int(sys.argv[4])
# select_every = int(sys.argv[5])
# warm_method = int(sys.argv[6])  # whether to use warmstart-onestep (1) or online (0)
# num_runs = int(sys.argv[7])    # number of random runs
learning_rate = 0.05
all_logs_dir = './results/debugging/' + data_name  + '/' + str(fraction) + '/' + str(select_every)
print(all_logs_dir)
subprocess.run(["mkdir", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
           '_selEvery:' + str(select_every) + '_variant' + str(warm_method) + '_runs' + str(num_runs)
print(exp_name)
exp_start_time = datetime.datetime.now()
print("=======================================", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)
if data_name == 'mnist':
    fullset, testset, num_cls = load_dataset_pytorch(datadir, data_name)
    # Validation Data set is 10% of the Entire Trainset.
    validation_set_fraction = 0.1
    num_fulltrn = len(fullset)
    num_val = int(num_fulltrn * validation_set_fraction)
    num_trn = num_fulltrn - num_val
    trainset, validset = random_split(fullset, [num_trn, num_val])
    x_trn, y_trn = trainset.dataset.data[trainset.indices], trainset.dataset.targets[trainset.indices]
    x_trn = torch.reshape(x_trn, [x_trn.shape[0], -1])
    x_val, y_val = validset.dataset.data[validset.indices], validset.dataset.targets[validset.indices]
    x_val = torch.reshape(x_val, [x_val.shape[0], -1])
    x_tst, y_tst = testset.data, testset.targets
    x_tst = torch.reshape(x_tst, [x_tst.shape[0], -1])
    trn_batch_size = 20
    val_batch_size = 20
    tst_batch_size = 20

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                              shuffle=False, pin_memory=True)

    valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                            shuffle=False, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                             shuffle=False, pin_memory=True)

print('-----------------------------------------')
print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
# print(y_trn.shape, y_val.shape, y_tst.shape)

N, M = x_trn.shape
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)
# Transfer all the data to GPU
d_t = time.time()
x_trn, y_trn = x_trn.to('cpu'), y_trn.to('cpu')
x_val, y_val = x_val.to('cpu'), y_val.to('cpu')
print("Transferred data to device in time:", time.time() - d_t)
print_every = 3


def random_greedy_train_model_online_taylor(start_rand_idxs, bud, lam):
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    idxs = start_rand_idxs
    total_idxs = list(np.arange(len(y_trn)))
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = SetFunction(trainloader, validset, model, criterion,
                             criterion_nored, learning_rate, device)
    print("Starting Randomized Greedy Online OneStep Run with taylor!")
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
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(lam * bud), clone_dict)
            rem_idxs = list(set(total_idxs).difference(set(subset_idxs)))
            subset_idxs.extend(list(np.random.choice(rem_idxs, size=int((1 - lam) * bud), replace=False)))
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(fullset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
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
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs


def train_model_online(start_rand_idxs, bud):
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    """
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    """
    model = model.to(device)
    idxs = start_rand_idxs
    total_idxs = list(np.arange(len(y_trn)))
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = SetFunction(trainloader, validset, model, criterion,
                             criterion_nored, learning_rate, device)
    print("Starting Greedy Online OneStep Run with taylor!")
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
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(bud), clone_dict)
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(fullset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
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
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs


def facloc_reg_train_model_online_taylor(start_rand_idxs, facloc_idxs, bud, lam):
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    val_plus_facloc_idxs = facloc_idxs.copy()
    val_plus_facloc_idxs.extend(validset.indices)
    wts = torch.ones(len(val_plus_facloc_idxs))
    wts[0:len(facloc_idxs)] = lam
    wts = wts.to(device)
    combined_set = torch.utils.data.Subset(fullset, val_plus_facloc_idxs)
    model = model.to(device)
    idxs = start_rand_idxs
    total_idxs = list(np.arange(len(y_trn)))
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = WtSetFunction(trainloader, combined_set, len(facloc_idxs), lam, model, criterion,
                             criterion_nored, learning_rate, device)
    print("Starting Facloc regularized Greedy Online OneStep Run with taylor!")
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
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(bud), clone_dict)
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(fullset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
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
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs


def train_model_mod_online(start_rand_idxs, bud):
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    """
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    """
    model = model.to(device)
    idxs = start_rand_idxs
    total_idxs = list(np.arange(len(y_trn)))
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = SetFunction(trainloader, validset, model, criterion,
                             criterion_nored, learning_rate, device)
    print("Starting Randomized Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # targets can have non_blocking=True.
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    for i in range(0, int(num_epochs/20)):
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
        if ((i + 1) % 5) == 0:
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(bud), clone_dict)
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(fullset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
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
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs


def train_model_random(start_rand_idxs):
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
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
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses


def train_model_random_online(start_rand_idxs):
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting Random Run")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)


    for i in range(0, num_epochs):
        subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                       sampler=SubsetRandomSampler(idxs),
                                                       pin_memory=True)
        idxs = np.random.choice(N, size=bud, replace=False)
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
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses


def run_stochastic_Facloc(data, targets, budget):
    model = MnistNet()
    model = model.to(device='cpu')
    approximate_error = 0.01
    per_iter_bud = 10
    num_iterations = int(budget/10)
    facloc_indices = []
    trn_indices = list(np.arange(len(data)))
    sample_size = int(len(data) / num_iterations * math.log(1 / approximate_error))
    #greedy_batch_size = 1200
    for i in range(num_iterations):
        rem_indices = list(set(trn_indices).difference(set(facloc_indices)))
        sub_indices = np.random.choice(rem_indices, size=sample_size, replace=False)
        data_subset = data[sub_indices].cpu()
        targets_subset = targets[sub_indices].cpu()
        train_loader_greedy = []
        train_loader_greedy.append((data_subset, targets_subset))
        setf_model = SetFunctionFacLoc(device, train_loader_greedy)
        idxs = setf_model.lazy_greedy_max(per_iter_bud, model)
        facloc_indices.extend([sub_indices[idx] for idx in idxs])
    return facloc_indices


def train_model_Facloc(idxs):
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting Facility Location Run")
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
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs



facloc_idxs = run_stochastic_Facloc(x_trn, y_trn, bud)
start_idxs = np.random.choice(N, size=bud, replace=False)
random_subset_idx = [x for x in start_idxs]

# Online algo run
start_time = time.time()
one_val_valacc, one_val_tstacc, one_val_subtrn_acc, one_val_full_trn_acc, one_val_valloss, \
one_val_tstloss,  one_val_subtrnloss, one_val_full_trn_loss, one_fval_val_losses, \
one_fval_substrn_losses, one_fval_fulltrn_losses, one_subset_idxs = \
    train_model_online(start_idxs, bud)
end_time = time.time()
one_step_time = end_time - start_time
print("One step run time: ", one_step_time)

# Random Run
start_time = time.time()
rand_valacc, rand_tstacc, rand_subtrn_acc, rand_full_trn_acc, rand_valloss, rand_tstloss, rand_subtrnloss,\
rand_full_trn_loss,rand_val_losses, rand_substrn_losses, rand_fulltrn_losses = train_model_random(start_idxs)
end_time = time.time()
random_run_time = end_time - start_time
print("Random Run Time: ", random_run_time)

#Online Random Run
start_time = time.time()
ol_rand_valacc, ol_rand_tstacc, ol_rand_subtrn_acc, ol_rand_full_trn_acc, ol_rand_valloss, ol_rand_tstloss, ol_rand_subtrnloss,\
ol_rand_full_trn_loss, ol_rand_val_losses, ol_rand_substrn_losses, ol_rand_fulltrn_losses = train_model_random_online(start_idxs)
end_time = time.time()
ol_random_run_time = end_time - start_time
print("Online Random Run Time: ", ol_random_run_time)

# Facility Location OneStep Runs
start_time = time.time()
facloc_reg_val_valacc, facloc_reg_val_tstacc, facloc_reg_val_subtrn_acc, facloc_reg_val_full_trn_acc, facloc_reg_val_valloss, \
facloc_reg_val_tstloss,  facloc_reg_val_subtrnloss, facloc_reg_val_full_trn_loss, facloc_reg_fval_val_losses, \
facloc_reg_fval_substrn_losses, facloc_reg_fval_fulltrn_losses, facloc_reg_subset_idxs = \
facloc_reg_train_model_online_taylor(start_idxs, facloc_idxs, bud, 100)
end_time = time.time()
facloc_one_step_time = end_time - start_time
print("Facility Location One step run time: ", facloc_one_step_time)

# Randomized Greedy Taylor OneStep Runs
start_time = time.time()
rand_reg_val_valacc, rand_reg_val_tstacc, rand_reg_val_subtrn_acc, rand_reg_val_full_trn_acc, rand_reg_val_valloss, \
rand_reg_val_tstloss,  rand_reg_val_subtrnloss,rand_reg_val_full_trn_loss, rand_reg_fval_val_losses, \
rand_reg_fval_substrn_losses, rand_reg_fval_fulltrn_losses, rand_reg_subset_idxs = \
    random_greedy_train_model_online_taylor(start_idxs, bud, 0.9)
end_time = time.time()
random_reg_one_step_time = end_time - start_time
print("Random Reg One Step run time: ", random_reg_one_step_time)

# Facility Location Run
start_time = time.time()
facloc_valacc, facloc_tstacc, facloc_subtrn_acc, facloc_full_tran_acc, facloc_valloss, facloc_tstloss, \
facloc_subtrnloss,facloc_full_trn_loss,facloc_val_losses, facloc_substrn_losses, facloc_fulltrn_losses, \
facloc_idxs = train_model_Facloc(facloc_idxs)
end_time = end_time
facloc_time = end_time - start_time
print("Facility location run time: ", facloc_time)

# Modified OneStep Runs
start_time = time.time()
mod_val_valacc, mod_val_tstacc, mod_val_subtrn_acc, mod_val_full_trn_acc, mod_val_valloss, mod_val_tstloss, \
mod_val_subtrnloss, mod_val_full_trn_loss, mod_val_val_losses, mod_val_substrn_losses, mod_val_fulltrn_losses,\
mod_subset_idxs = train_model_mod_online(start_idxs, bud)
end_time = time.time()
mod_one_step_time = end_time - start_time
print("Mod One Step run time: ", mod_one_step_time)

plot_start_epoch = 0
###### Subset Trn loss with val = VAL #############
plt.figure()
# plt.plot(np.arange(1,num_epochs+1), knn_fval_substrn_losses, 'r-', label='knn_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), one_fval_substrn_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_substrn_losses[plot_start_epoch:], 'g-', label='random')
plt.plot(np.arange(plot_start_epoch, num_epochs), ol_rand_substrn_losses[plot_start_epoch:], 'g+', label='Online random')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_substrn_losses[plot_start_epoch:], 'pink', label='FacLoc')
# plt.plot(np.arange(plot_start_epoch,num_epochs), mod_tay_fval_substrn_losses[plot_start_epoch:], 'c', label='after_opt_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_reg_fval_substrn_losses[plot_start_epoch:], 'k-',
         label='rand_reg_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_fval_substrn_losses[plot_start_epoch:], 'y',
         label='facloc_reg_tay_v=val')
# plt.plot(np.arange(1,num_epochs+1), fc_substrn_losses, '#000000', label='Facility Location')
# plt.plot(np.arange(1,num_epochs+1), nontay_fval_substrn_losses, 'orange', label='NON-tay_v=val')


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
plt.plot(np.arange(plot_start_epoch, num_epochs), one_fval_fulltrn_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_fulltrn_losses[plot_start_epoch:], 'g-', label='random')
plt.plot(np.arange(plot_start_epoch, num_epochs), ol_rand_fulltrn_losses[plot_start_epoch:], 'g+', label='Online random')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_fulltrn_losses[plot_start_epoch:], 'pink', label='FacLoc')
# plt.plot(np.arange(plot_start_epoch, num_epochs), mod_tay_fval_fulltrn_losses[plot_start_epoch:], 'c', label='after_opt_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_reg_fval_fulltrn_losses[plot_start_epoch:], 'k-',
         label='rand_reg_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_fval_fulltrn_losses[plot_start_epoch:], 'y',
         label='facloc_reg_tay_v=val')

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
plt.plot(np.arange(plot_start_epoch, num_epochs), one_fval_val_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_val_losses[plot_start_epoch:], 'g-', label='random')
plt.plot(np.arange(plot_start_epoch, num_epochs), ol_rand_val_losses[plot_start_epoch:], 'g+', label='Online random')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_val_losses[plot_start_epoch:], 'pink', label='FacLoc')
#plt.plot(np.arange(plot_start_epoch,num_epochs), mod_tay_fval_val_losses[plot_start_epoch:], 'c', label='after_opt_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_reg_fval_val_losses[plot_start_epoch:], 'k-',
         label='rand_reg_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_fval_val_losses[plot_start_epoch:], 'y',
         label='facloc_reg_tay_v=val')

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
print('*| Facility Location             |', facloc_valacc, '  | ', facloc_tstacc, ' |', file=logfile)
print('*| Taylor with Validation=VAL     |', one_val_valacc, '  | ', one_val_tstacc, ' |', file=logfile)
print('*| Random Selection               |', rand_valacc, '  | ', rand_tstacc, ' |', file=logfile)
print('*| Online Random Selection               |', ol_rand_valacc, '  | ', ol_rand_tstacc, ' |', file=logfile)
print('*| Taylor after training               |', mod_val_valacc, '  | ', mod_val_tstacc, ' |', file=logfile)
print('*| random regularized Taylor after training               |', rand_reg_val_valacc, '  | ', rand_reg_val_tstacc, ' |',
      file=logfile)
print('*| facloc regularizec Taylor after training               |', facloc_reg_val_valacc, '  | ',
      facloc_reg_val_tstacc, ' |', file=logfile)
print('---------------------------------------------------', file=logfile)
print('|Algo                            | Run Time       |', file=logfile)
print('| -------------------------------|:-------------:|', file=logfile)
print('*| Facility Location             |', facloc_time, '  | ', file=logfile)
print('*| Taylor with Validation=VAL     |', one_step_time, '  | ', file=logfile)
print('*| Random Selection               |', random_run_time, '  | ',file=logfile)
print('*| Online Random Selection               |', ol_random_run_time, '  | ',file=logfile)
print('*| Taylor after training               |', mod_one_step_time, '  | ', file=logfile)
print('*| random regularized Taylor after training               |', random_reg_one_step_time,' |',
      file=logfile)
print('*| facloc regularizec Taylor after training               |', facloc_one_step_time, '  | ',
       file=logfile)

print("\n", file=logfile)

print("=========Random Results==============", file=logfile)
print("*Rand Validation LOSS:", rand_valloss, file=logfile)
print("*Rand Test Data LOSS:", rand_tstloss, file=logfile)
print("*Rand Full Trn Data LOSS:", rand_fulltrn_losses[-1], file=logfile)

print("=========Online Random Results==============", file=logfile)
print("*Rand Validation LOSS:", ol_rand_valloss, file=logfile)
print("*Rand Test Data LOSS:", ol_rand_tstloss, file=logfile)
print("*Rand Full Trn Data LOSS:", ol_rand_fulltrn_losses[-1], file=logfile)

print("=========FacLoc Results==============", file=logfile)
print("*Facloc Validation LOSS:", facloc_valloss, file=logfile)
print("*Facloc Test Data LOSS:", facloc_tstloss, file=logfile)
print("*Facloc Full Trn Data LOSS:", facloc_fulltrn_losses[-1], file=logfile)

print("=========Online Selection Taylor with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", one_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", one_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", one_fval_fulltrn_losses[-1], file=logfile)

print("=========Random Regularized Online Selection Taylor with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", rand_reg_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", rand_reg_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", rand_reg_fval_fulltrn_losses[-1], file=logfile)

print("=========Facility Location Loss regularized Online Selection Taylor with Validation Set===================",
      file=logfile)
print("*Taylor v=VAL Validation LOSS:", facloc_reg_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", facloc_reg_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", facloc_reg_fval_fulltrn_losses[-1], file=logfile)

print("=========Online Selection Taylor after model training with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", mod_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", mod_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", mod_val_fulltrn_losses[-1], file=logfile)
print("=============================================================================================", file=logfile)
print("---------------------------------------------------------------------------------------------", file=logfile)
print("\n", file=logfile)

mod_subset_idxs = list(mod_subset_idxs)
with open(all_logs_dir + '\mod_one_step_subset_selected.txt', 'w') as log_file:
    print(mod_subset_idxs, file=log_file)

subset_idxs = list(one_subset_idxs)
with open(all_logs_dir + '\one_step_subset_selected.txt', 'w') as log_file:
    print(subset_idxs, file=log_file)

rand_subset_idxs = list(rand_reg_subset_idxs)
with open(all_logs_dir + '\rand_reg_one_step_subset_selected.txt', 'w') as log_file:
    print(rand_subset_idxs, file=log_file)

facloc_reg_subset_idxs = list(facloc_reg_subset_idxs)
with open(all_logs_dir + '\facloc_reg_one_step_subset_selected.txt', 'w') as log_file:
    print(facloc_reg_subset_idxs, file=log_file)

random_subset_idx = list(random_subset_idx)
with open(all_logs_dir + '\random_subset_selected.txt', 'w') as log_file:
    print(random_subset_idx, file=log_file)

facloc_idxs = list(facloc_idxs)
with open(all_logs_dir + '\facloc_subset_selected.txt', 'w') as log_file:
    print(facloc_idxs, file=log_file)

