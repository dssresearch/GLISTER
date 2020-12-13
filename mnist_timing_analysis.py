import copy
import numpy as np
import os
import subprocess
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from matplotlib import pyplot as plt
from models.simpleNN_net import *  # ThreeLayerNet
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler
from models.set_function_all import SetFunctionFacLoc, SetFunctionBatch  # as SetFunction #SetFunctionCompare
from models.set_function_grad_computation_taylor import NonDeepSetRmodularFunction as SetFunctionTaylor
from models.set_function_craig import PerClassNonDeepSetFunction as CRAIG
from models.set_function_ideas import SetFunctionTaylorDeep_ReLoss_Mean
from sklearn.model_selection import train_test_split
from utils.custom_dataset import load_dataset_custom, load_mnist_cifar
import math

def train_model_craig(start_rand_idxs, bud, convex=True, every=False):
    torch.manual_seed(42)
    np.random.seed(42)
    # model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # model = ThreeLayerNet(M, num_cls, 5, 5)
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Starting CRAIG Algorithm!")
    eta = 0.01
    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    #device, x_trn, y_trn, model, N_trn, batch_size, if_convex)
    setf = CRAIG(device, x_trn, y_trn, model, N, 1000, False)
    cached_state_dict = copy.deepcopy(model.state_dict())
    clone_dict = copy.deepcopy(model.state_dict())
    idxs, gammas = setf.lazy_greedy_max(bud, clone_dict)
    model.load_state_dict(cached_state_dict)
    exp_start_time_craig = time.time()
    for i in range(num_epochs):
        print(i)
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        start_time = time.time()
        inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        optimizer.zero_grad()
        scores = model(inputs)
        losses = criterion_nored(scores, targets)
        loss = torch.dot(torch.from_numpy(np.array(gammas)).to(device).type(torch.float) / N, losses)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predict = val_outputs.max(1)
            val_correct = val_predict.eq(y_val).sum().item()
            val_total = y_val.size(0)
            val_accu = 100 * val_correct / val_total

            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)
            _, full_predict = full_trn_outputs.max(1)
            full_correct = full_predict.eq(y_trn).sum().item()
            full_total = y_trn.size(0)
            full_acc = 100 * full_correct / full_total

            tst_outputs = model(x_tst)
            tst_loss = criterion(tst_outputs, y_tst)
            _, tst_predict = tst_outputs.max(1)
            tst_correct = tst_predict.eq(y_tst).sum().item()
            tst_total = y_tst.size(0)
            tst_accu = 100 * tst_correct / tst_total

        timing[i] = time.time() - start_time
        print(timing[i])
        val_acc[i] = val_accu
        tst_acc[i] = tst_accu
        if not convex:
            if not every and (i + 1) % select_every == 0:
                cached_state_dict = copy.deepcopy(model.state_dict())
                clone_dict = copy.deepcopy(model.state_dict())
                idxs, gammas = setf.lazy_greedy_max(bud, clone_dict)
                model.load_state_dict(cached_state_dict)
                # gammas = gammas.type(torch.float)/N
            else:
                cached_state_dict = copy.deepcopy(model.state_dict())
                clone_dict = copy.deepcopy(model.state_dict())
                idxs, gammas = setf.lazy_greedy_max(bud, clone_dict)
                model.load_state_dict(cached_state_dict)
                # gammas = gammas.type(torch.float)/N
    time_list = timing
    print("CRAIG", file=logfile)
    print('---------------------------------------------------------------------', file=logfile)
    val = "Validation Accuracy,"
    tst = "Test Accuracy,"
    timing = "Time,"
    for i in range(num_epochs):
        timing = timing + "," + str(time_list[i])
        val = val + "," + str(val_acc[i])
        tst = tst + "," + str(tst_acc[i])
    print(timing, file=logfile)
    print(val, file=logfile)
    print(tst, file=logfile)

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_accu)
    print("Test Data Loss and Accuracy:", tst_loss.item(), tst_accu)
    print('-----------------------------------')
    return val_acc, tst_acc, time_list


def train_model_taylor(func_name, r, start_rand_idxs=None, bud=None):
    torch.manual_seed(42)
    np.random.seed(42)
    model = TwoLayerNet(M, num_cls, 100)
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    setf_model = SetFunctionTaylor(trainset, x_val, y_val, model, criterion,
                                   criterion_nored, learning_rate, device, num_cls, 1000)

    if func_name == 'Taylor Online':
        print("Starting Online OneStep Run with taylor on loss!")
    timing = np.zeros(num_epochs)
    substrn_grads = []
    tst_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    exp_start_time_onestep = time.time()
    for i in range(num_epochs):
        start_time = time.time()
        if ((i + 1) % select_every == 0) and func_name not in ['Facility Location', 'Random', "KNNSB"]:
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            t_ng_start = time.time()
            new_idxs = setf_model.naive_greedy_max(bud, r, clone_dict)  # , grads_idxs
            t_ng_end = time.time() - t_ng_start
            print("Subset Selection Time: " + str(t_ng_end))
            idxs = new_idxs  # update the current set
            model.load_state_dict(cached_state_dict)

        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()
        timing[i] = time.time() - start_time
        print(timing[i])
        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predict = val_outputs.max(1)
            val_correct = val_predict.eq(y_val).sum().item()
            val_total = y_val.size(0)
            val_accu = 100 * val_correct / val_total

            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)
            _, full_predict = full_trn_outputs.max(1)
            full_correct = full_predict.eq(y_trn).sum().item()
            full_total = y_trn.size(0)
            full_acc = 100 * full_correct / full_total

            tst_outputs = model(x_tst)
            tst_loss = criterion(tst_outputs, y_tst)
            _, tst_predict = tst_outputs.max(1)
            tst_correct = tst_predict.eq(y_tst).sum().item()
            tst_total = y_tst.size(0)
            tst_accu = 100 * tst_correct / tst_total

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())
        val_acc[i] = val_accu
        tst_acc[i] = tst_accu
    time_list = timing
    print("One Step Training set", file=logfile)
    print('---------------------------------------------------------------------', file=logfile)
    val = "Validation Accuracy,"
    tst = "Test Accuracy,"
    timing = "Time,"
    for i in range(num_epochs):
        timing = timing + "," + str(time_list[i])
        val = val + "," + str(val_acc[i])
        tst = tst + "," + str(tst_acc[i])
    print(timing, file=logfile)
    print(val, file=logfile)
    print(tst, file=logfile)

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_accu)
    print("Test Data Loss and Accuracy:", tst_loss.item(), tst_accu)
    print('-----------------------------------')
    return val_acc, tst_acc, time_list


def train_model_mod_taylor(start_rand_idxs, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    # model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # model = ThreeLayerNet(M, num_cls, 5, 5)
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Starting Online OneStep Run after optimization with taylor!")

    timing = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    # idxs = start_rand_idxs
    batch_size = 20
    exp_start_time_full = time.time()
    for i in range(num_epochs):
        start_time = time.time()
        batch_wise_indices = list(BatchSampler(RandomSampler(trainset), bud, drop_last=False))
        for batch_idx in batch_wise_indices:
            # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
            inputs, targets = x_trn[batch_idx], y_trn[batch_idx]
            optimizer.zero_grad()
            scores = model(inputs)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
        timing[i] = time.time() - start_time
        print(timing[i])
        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predict = val_outputs.max(1)
            val_correct = val_predict.eq(y_val).sum().item()
            val_total = y_val.size(0)
            val_accu = 100 * val_correct / val_total

            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)
            _, full_predict = full_trn_outputs.max(1)
            full_correct = full_predict.eq(y_trn).sum().item()
            full_total = y_trn.size(0)
            full_acc = 100 * full_correct / full_total

            tst_outputs = model(x_tst)
            tst_loss = criterion(tst_outputs, y_tst)
            _, tst_predict = tst_outputs.max(1)
            tst_correct = tst_predict.eq(y_tst).sum().item()
            tst_total = y_tst.size(0)
            tst_accu = 100 * tst_correct / tst_total
        val_acc[i] = val_accu
        tst_acc[i] = tst_accu
    time_list = timing
    print("Full Training set", file=logfile)
    print('---------------------------------------------------------------------', file=logfile)
    val = "Validation Accuracy,"
    tst = "Test Accuracy,"
    timing = "Time,"
    for i in range(num_epochs):
        timing = timing + "," + str(time_list[i])
        val = val + "," + str(val_acc[i])
        tst = tst + "," + str(tst_acc[i])
    print(timing, file=logfile)
    print(val, file=logfile)
    print(tst, file=logfile)
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_accu)
    print("Test Data Loss and Accuracy:", tst_loss.item(), tst_accu)
    print('-----------------------------------')

    return val_acc, tst_acc, time_list


device = "cpu"
print("Using Device:", device)

datadir = '../data/' + 'dna/'
data_name = 'dna'
fraction = 0.1
num_epochs = 200
select_every = 20
feature = 'DSS'
warm_method = 0  # whether to use warmstart-onestep (1) or online (0)
num_runs = 1  # number of random runs
learning_rate = 0.05

if data_name in ['dna', 'sklearn-digits', 'satimage', 'svmguide1', 'letter', 'shuttle', 'ijcnn1', 'sensorless',
                 'connect_4', 'sensit_seismic', 'usps']:
    fullset, valset, testset, data_dims, num_cls = load_dataset_custom(datadir, data_name, feature=feature, isnumpy=False)
    validation_set_fraction = 0.1
    num_fulltrn = len(fullset)
    num_val = int(num_fulltrn * validation_set_fraction)
    num_trn = num_fulltrn - num_val
    trainset, validset = random_split(fullset, [num_trn, num_val])
elif data_name in ['mnist', "fashion-mnist", 'cifar10']:
    fullset, valset, testset, num_cls = load_mnist_cifar(datadir, data_name, feature=feature)
    validation_set_fraction = 0.1
    num_fulltrn = len(fullset)
    num_val = int(num_fulltrn * validation_set_fraction)
    num_trn = num_fulltrn - num_val
    trainset, validset = random_split(fullset, [num_trn, num_val])

cnt = 0
batch_wise_indices = list(BatchSampler(SequentialSampler(trainset), 1000, drop_last=False))
for batch_idx in batch_wise_indices:
    inputs = torch.cat([trainset[x][0].view(1, -1) for x in batch_idx],
                       dim=0).type(torch.float)
    targets = torch.tensor([trainset[x][1] for x in batch_idx])
    if cnt == 0:
        x_trn = inputs
        y_trn = targets
        cnt = cnt + 1
    else:
        x_trn = torch.cat([x_trn, inputs], dim=0)
        y_trn = torch.cat([y_trn, targets], dim=0)
        cnt = cnt + 1
batch_wise_indices = list(BatchSampler(SequentialSampler(valset), 1000, drop_last=False))
for batch_idx in batch_wise_indices:
    inputs = torch.cat([valset[x][0].view(1, -1) for x in batch_idx],
                       dim=0).type(torch.float)
    targets = torch.tensor([valset[x][1] for x in batch_idx])
    if cnt == 0:
        x_val = inputs
        y_val = targets
        cnt = cnt + 1
    else:
        x_val = torch.cat([x_trn, inputs], dim=0)
        y_val = torch.cat([y_trn, targets], dim=0)
        cnt = cnt + 1
batch_wise_indices = list(BatchSampler(SequentialSampler(testset), 1000, drop_last=False))
# x_trn, y_trn = fullset.data, fullset.targets
for batch_idx in batch_wise_indices:
    inputs = torch.cat([testset[x][0].view(1, -1) for x in batch_idx],
                       dim=0).type(torch.float)
    targets = torch.tensor([testset[x][1] for x in batch_idx])
    if cnt == 0:
        x_tst = inputs
        y_tst = targets
        cnt = cnt + 1
    else:
        x_tst = torch.cat([x_trn, inputs], dim=0)
        y_tst = torch.cat([y_trn, targets], dim=0)
        cnt = cnt + 1

# x_tst, y_tst = testset.data, testset.targets

x_trn = x_trn.view(x_trn.shape[0], -1)
x_val = x_val.view(x_val.shape[0], -1)
x_tst = x_tst.view(x_tst.shape[0], -1)
# Get validation data: Its 10% of the entire (full) training data


print('-----------------------------------------')
# print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
# print(y_trn.shape, y_val.shape, y_tst.shape)

N, M = x_trn.shape
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)
# Transfer all the data to GPU
d_t = time.time()
x_trn, y_trn = x_trn.to(device), y_trn.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)
x_tst, y_tst = x_tst.to(device), y_tst.to(device)
print("Transferred data to device in time:", time.time() - d_t)
print_every = 1
train_batch_size_for_greedy = 1200
train_loader_greedy = []
for item in range(math.ceil(len(x_trn) / train_batch_size_for_greedy)):
    inputs = x_trn[item * train_batch_size_for_greedy:(item + 1) * train_batch_size_for_greedy]
    target = y_trn[item * train_batch_size_for_greedy:(item + 1) * train_batch_size_for_greedy]
    train_loader_greedy.append((inputs, target))

valid_loader = []
train_batch_size = 128
for item in range(math.ceil(len(x_val) / train_batch_size)):
    inputs = x_val[item * train_batch_size:(item + 1) * train_batch_size]
    target = y_val[item * train_batch_size:(item + 1) * train_batch_size]
    valid_loader.append((inputs, target))

train_loader = []
for item in range(math.ceil(len(x_trn) / train_batch_size)):
    inputs = x_trn[item * train_batch_size:(item + 1) * train_batch_size]
    target = y_trn[item * train_batch_size:(item + 1) * train_batch_size]
    train_loader.append((inputs, target))

start_idxs = np.random.choice(N, size=bud, replace=False)
random_subset_idx = [x for x in start_idxs]

R = [1, 5, 10, 20]
# CRAIG Run
# print(time.time())
# craig_valacc, craig_tstacc, craig_timing = train_model_craig(start_idxs, bud, False, False)
# print(time.time())
for r in R:
    all_logs_dir = './results/timing' + data_name + '/adam/' + str(fraction) + '/' + str(r) + '/' + str(select_every)
    print(all_logs_dir)
    subprocess.run(["mkdir", "-p", all_logs_dir])
    path_logfile = os.path.join(all_logs_dir, data_name + 'craig' + '.txt')
    logfile = open(path_logfile, 'w')
    exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
               '_selEvery:' + str(select_every) + '_variant' + str(warm_method) + '_runs' + str(num_runs)
    print(exp_name)
    # Online algo run
    #craig_valacc, craig_tstacc, craig_timing = train_model_craig(start_idxs, bud, False, False)

    print(time.time())
    t_val_valacc, t_val_tstacc, t_val_timing = train_model_taylor('Taylor Online', r, start_idxs, bud)
    print(time.time())

    r_val_valacc, r_val_tstacc, r_val_timing = train_model_taylor('Random', r, start_idxs, bud)
    #Full Data Training
    print(time.time())
    mod_t_valacc, mod_t_tstacc, mod_t_timing = train_model_mod_taylor(start_idxs, bud, True)
    print(time.time())

    sum = 0
    mod_cumulative_timing = np.zeros(len(mod_t_timing))
    for i in range(len(mod_cumulative_timing)):
        sum += mod_t_timing[i]
        mod_cumulative_timing[i] = sum
    mod_t_timing = mod_cumulative_timing

    sum = 0
    t_val_cumulative_timing = np.zeros(len(t_val_timing))
    for i in range(len(t_val_timing)):
        sum += t_val_timing[i]
        t_val_cumulative_timing[i] = sum
    t_val_timing = t_val_cumulative_timing

    sum = 0
    r_val_cumulative_timing = np.zeros(len(r_val_timing))
    for i in range(len(r_val_timing)):
        sum += r_val_timing[i]
        r_val_cumulative_timing[i] = sum
    r_val_timing = r_val_cumulative_timing
    ###### Test accuray #############

    plt.figure()
    #plt.plot(craig_timing, craig_tstacc, 'g-', label='CRAIG')
    plt.plot(mod_t_timing, mod_t_tstacc, 'orange', label='Full training')
    plt.plot(t_val_timing, t_val_tstacc, 'b-', label='GLISTER')
    plt.plot(r_val_timing, r_val_tstacc, 'r', label='Random')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Test accuracy')
    plt.title('Test Accuracy vs Time ' + data_name + '_' + str(fraction))
    plt_file = path_logfile + '_' + str(fraction) + 'tst_accuracy_v=VAL.png'
    plt.savefig(plt_file)
    plt.clf()

    #craig_cumulative_timing = [0 for x in craig_timing]
    #for i in range(len(craig_timing)):
    #    craig_cumulative_timing[i] = np.array(craig_timing)[0:i+1].sum()
    #craig_timing = craig_cumulative_timing


    ###### Validation #############

    plt.figure()
    #plt.plot(craig_timing, craig_valacc, 'g-', label='CRAIG')
    plt.plot(mod_t_timing, mod_t_valacc, 'orange', label='Full training')
    plt.plot(t_val_timing, t_val_valacc, 'b-', label='GLISTER')
    plt.plot(r_val_timing, r_val_valacc, 'r', label='Random')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Validation accuracy')
    plt.title('Validation Accuracy vs Time ' + data_name + '_' + str(fraction))
    plt_file = path_logfile + '_' + str(fraction) + 'val_accuracy_v=VAL.png'
    plt.savefig(plt_file)
    plt.clf()

