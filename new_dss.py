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
from models.simpleNN_net import *  # ThreeLayerNet
from models.logistic_regression import LogisticRegNet
from models.set_function_all import SetFunctionFacLoc, SetFunctionTaylor, \
    SetFunctionBatch  # as SetFunction #SetFunctionCompare
from models.set_function_grad_computation_taylor import Small_Glister_Linear_SetFunction_Closed, Small_GLISTER_WeightedSetFunction
from models.set_function_craig import SetFunction2 as CRAIG
from sklearn.model_selection import train_test_split
from utils.custom_dataset import load_dataset_custom, load_mnist_cifar, write_knndata
import math
import random
from torch.utils.data import TensorDataset, DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# #device = "cpu"
print("Using Device:", device)

## Convert to this argparse
datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
select_every = int(sys.argv[5])
feature = sys.argv[6]  # 70
warm_method = 0  # whether to use warmstart-onestep (1) or online (0)
num_runs = 1  # number of random runs
learning_rate = 0.05
all_logs_dir = './results/NN/' + data_name + '/' + feature + '/' + str(fraction) + '/' + str(select_every)
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

# print("Datadir",datadir,"Data name",data_name)

# write_knndata(datadir, data_name)

if data_name == 'mnist':

    fullset, valset, testset, num_cls = load_mnist_cifar(datadir, data_name, feature=feature)

    x_trn, y_trn = fullset.data, fullset.targets
    x_tst, y_tst = testset.data, testset.targets
    x_trn = x_trn.view(x_trn.shape[0], -1)
    x_tst = x_tst.view(x_tst.shape[0], -1)
    # Get validation data: Its 10% of the entire (full) training data
    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
else:

    fullset, valset, testset, data_dims, num_cls = load_dataset_custom(datadir, data_name, feature=feature,
                                                                       isnumpy=True)

    x_trn, y_trn = torch.from_numpy(fullset[0]).float(), torch.from_numpy(fullset[1]).long()
    x_tst, y_tst = torch.from_numpy(testset[0]).float(), torch.from_numpy(testset[1]).long()
    x_val, y_val = torch.from_numpy(valset[0]).float(), torch.from_numpy(valset[1]).long()

write_knndata(datadir, data_name, feature=feature)

print('-----------------------------------------')
print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
# print(y_trn.shape, y_val.shape, y_tst.shape)
np.random.seed(42)
N, M = x_trn.shape
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)
# Transfer all the data to GPU
d_t = time.time()
x_trn, y_trn = x_trn.to(device), y_trn.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)
print("Transferred data to device in time:", time.time() - d_t)
print_every = 50
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


def perform_knnsb_selection(datadir, dset_name, budget, selUsing):
    run_path = './knn_results/'
    output_dir = run_path + 'KNNSubmod_' + dset_name + '/'
    indices_file = output_dir + feature + '_KNNSubmod_' + str((int)(budget * 100)) + ".subset"

    if os.path.exists(indices_file):
        idxs_knnsb = np.genfromtxt(indices_file, delimiter=',', dtype=int)  # since they are indices!
        return idxs_knnsb

    trn_filepath = os.path.join(datadir, feature + '_knn_' + dset_name + '.trn')

    if selUsing == 'val':
        val_filepath = os.path.join(datadir, feature + 'knn_' + dset_name + '.val')
    else:
        val_filepath = trn_filepath

    subprocess.call(["mkdir", "-p", output_dir])
    knnsb_args = []
    knnsb_args.append('./build/KNNSubmod')
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
    idxs_knnsb = np.genfromtxt(indices_file, delimiter=',', dtype=int)  # since they are indices!
    return idxs_knnsb


def gen_rand_prior_indices(size):
    per_sample_count = [len(torch.where(y_trn == x)[0]) for x in np.arange(num_cls)]
    per_sample_budget = int(size / num_cls)
    total_set = list(np.arange(N))
    indices = []
    count = 0
    for i in range(num_cls):
        label_idxs = torch.where(y_trn == i)[0].cpu().numpy()
        if per_sample_count[i] > per_sample_budget:
            indices.extend(list(np.random.choice(label_idxs, size=per_sample_budget, replace=False)))
        else:
            indices.extend(label_idxs)
            count += (per_sample_budget - per_sample_count[i])
    for i in indices:
        total_set.remove(i)
    indices.extend(list(np.random.choice(total_set, size=count, replace=False)))
    return indices


def train_model_craig(start_rand_idxs, bud, convex=True, every=False):
    torch.manual_seed(42)
    np.random.seed(42)
    model = LogisticRegNet(M, num_cls)
    # model = TwoLayerNet(M, num_cls, 100)
    # model = ThreeLayerNet(M, num_cls, 5, 5)
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting CRAIG Algorithm!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    setf = CRAIG(device, train_loader, True)
    # idxs = start_rand_idxs
    idxs, gammas = setf.lazy_greedy_max(bud, model)
    for i in range(num_epochs):
        inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        optimizer.zero_grad()
        scores = model(inputs)
        losses = criterion_nored(scores, targets)
        loss = torch.dot(torch.from_numpy(np.array(gammas)).to(device).type(torch.float), losses)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn)
        substrn_losses[i] = losses.mean().item()
        fulltrn_losses[i] = full_trn_loss.item()
        val_losses[i] = val_loss.item()
        if not convex:
            if not every and (i + 1) % select_every == 0:
                idxs, gammas = setf.lazy_greedy_max(bud, model)
            else:
                idxs, gammas = setf.lazy_greedy_max(bud, model)
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


def train_model_taylor(func_name, start_rand_idxs=None, bud=None, valid=True, fac_loc_idx=None):
    torch.manual_seed(42)
    np.random.seed(42)
    # model = ThreeLayerNet(M, num_cls, 100, 100)
    model = LogisticRegNet(M, num_cls)
    # model = TwoLayerNet(M, num_cls, 100)
    model = model.to(device)
    idxs = start_rand_idxs
    if func_name == 'Facloc Regularized':
        x_val1 = torch.cat([x_trn[fac_loc_idx], x_val], dim=0)
        y_val1 = torch.cat([y_trn[fac_loc_idx], y_val], dim=0)
    total_idxs = list(np.arange(len(y_trn)))
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if func_name == 'Full OneStep':
        setf_model = SetFunctionBatch(x_trn, y_trn, x_val, y_val, valid, model,
                                      criterion, criterion_nored, learning_rate, device)
    elif func_name == 'Facility Location':
        setf_model = SetFunctionFacLoc(device, train_loader_greedy)
        idxs = setf_model.lazy_greedy_max(bud, model)
    elif func_name == 'Facloc Regularized':
        setf_model =  Small_GLISTER_WeightedSetFunction(x_trn, y_trn, x_val, y_val, len(fac_loc_idx), 10, model, criterion,
                 criterion_nored, learning_rate, device, num_cls)
    else:
        setf_model = Small_Glister_Linear_SetFunction_Closed(x_trn, y_trn, x_val, y_val, model, criterion,
                                                      criterion_nored, learning_rate, device, num_cls)

    if func_name == 'Taylor Online':
        print("Starting Online OneStep Run with taylor on loss!")
    elif func_name == 'Full OneStep':
        print("Starting Online OneStep Run without taylor!")
    elif func_name == 'Facloc Regularized':
        print("Starting Facility Location Regularized Online OneStep Run with taylor!")
    elif func_name == 'Random Greedy':
        print("Starting Randomized Greedy Online OneStep Run with taylor!")
    elif func_name == 'Online Random':
        print("Starting Random Selection with taylor!")
    elif func_name == 'Facility Location':
        print("Starting Facility Location!")
    elif func_name == 'KnnSB':
        print("Starting Supervised Facility Location!")
    elif func_name == 'Random':
        print("Starting Random Run!")
    elif func_name == 'Random with Prior':
        print("Starting Random with Prior Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    substrn_grads = []
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    # idxs = start_rand_idxs
    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        temp1 = torch.autograd.grad(loss, model.parameters())
        grad_value = torch.norm(torch.cat((temp1[0], temp1[1].view(-1, 1)), dim=1).flatten()).item()
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

        if ((i + 1) % select_every == 0) and func_name not in ['Facility Location', 'Random', 'Random with Prior',
                                                               'KNNSB']:
            substrn_grads.append(grad_value)
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            t_ng_start = time.time()

            if func_name == 'Random Greedy':
                new_idxs = setf_model.naive_greedy_max(int(0.9 * bud), clone_dict)
                rem_idxs = list(set(total_idxs).difference(set(new_idxs)))
                new_idxs.extend(list(np.random.choice(rem_idxs, size=int(0.1 * bud), replace=False)))
                idxs = new_idxs

            elif func_name == 'Online Random':
                state = np.random.get_state()
                np.random.seed(i)
                idxs = np.random.choice(N, size=bud, replace=False)
                np.random.set_state(state)

            else:
                new_idxs = setf_model.naive_greedy_max(bud, clone_dict)  # , grads_idxs
                idxs = new_idxs  # update the current set
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

    # classes,count = torch.unique(val_predict,return_counts=True)
    # print(count)

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

    # classes, count = torch.unique(predicted,return_counts=True)
    # print(count)
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc)
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item(), substrn_losses, fulltrn_losses, val_losses, idxs, substrn_grads


def train_model_mod_taylor(start_rand_idxs, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    model = LogisticRegNet(M, num_cls)
    # model = TwoLayerNet(M, num_cls, 100)
    # model = ThreeLayerNet(M, num_cls, 5, 5)
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting Online OneStep Run after optimization with taylor!")
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


start_idxs = np.random.choice(N, size=bud, replace=False)
random_subset_idx = [x for x in start_idxs]
rand_prior_idxs = gen_rand_prior_indices(size=bud)

# Online algo run
t_val_valacc, t_val_tstacc, t_val_valloss, t_val_tstloss, tay_fval_substrn_losses, tay_fval_fulltrn_losses, tay_fval_val_losses, subset_idxs, tay_grads = train_model_taylor(
    'Taylor Online', start_idxs, bud, True)

# Facility Location Run
fv1, ft1, fv2, ft2, facloc_substrn_losses, facloc_fulltrn_losses, facloc_val_losses, facloc_idxs, facloc_grads = train_model_taylor(
    'Facility Location', None, bud)

# Facility Location OneStep Runs
facloc_reg_t_val_valacc, facloc_reg_t_val_tstacc, facloc_reg_t_val_valloss, facloc_reg_t_val_tstloss, facloc_reg_tay_fval_substrn_losses, facloc_reg_tay_fval_fulltrn_losses,\
facloc_reg_tay_fval_val_losses, facloc_reg_subset_idxs, facloc_reg_grads = train_model_taylor(
    'Facloc Regularized', start_idxs, bud, True, facloc_idxs)

# CRAIG Run
craig_valacc, craig_tstacc, craig_valloss, craig_tstloss, craig_substrn_losses, craig_fulltrn_losses, \
craig_val_losses, craig_subset_idxs = train_model_craig(start_idxs, bud, True, False)

# Random Run
rv1, rt1, rv2, rt2, rand_substrn_losses, rand_fulltrn_losses, rand_val_losses, idxs, rand_grads = train_model_taylor('Random', start_idxs)

# Random with prior Run
if feature == 'classimb':
    rpv1, rpt1, rpv2, rpt2, rand_prior_substrn_losses, rand_prior_fulltrn_losses, rand_prior_val_losses, prior_idxs, \
    rand_prior_grads = train_model_taylor('Random with Prior', rand_prior_idxs)

## KnnSB selection with Flag = TRN and FLAG = VAL
#knn_idxs_flag_trn = perform_knnsb_selection(datadir, data_name, fraction, selUsing='trn')
#knn_idxs_flag_val = perform_knnsb_selection(datadir, data_name, fraction, selUsing='val')

## Training with KnnSB idxs with Flag = TRN and FLAG = VAL
#knn_valacc_flagtrn, knn_tstacc_flagtrn, knn_valloss_flagtrn, knn_tstloss_flagtrn, knn_ftrn_substrn_losses, knn_ftrn_fulltrn_losses, \
#knn_ftrn_val_losses, knn_ftrn_idxs, knn_ftrn_grads = train_model_taylor("KNNSB", knn_idxs_flag_trn)

#knn_valacc_flagval, knn_tstacc_flagval, knn_valloss_flagval, knn_tstloss_flagval, knn_fval_substrn_losses, knn_fval_fulltrn_losses, \
#knn_fval_val_losses, knn_fval_idxs, knn_fval_grads = train_model_taylor("KNNSB", knn_idxs_flag_val)
# Randomized Greedy Taylor OneStep Runs
rand_t_val_valacc, rand_t_val_tstacc, rand_t_val_valloss, rand_t_val_tstloss, rand_tay_fval_substrn_losses, rand_tay_fval_fulltrn_losses, rand_tay_fval_val_losses, rand_subset_idxs, rand_reg_grads = train_model_taylor(
    'Random Greedy', start_idxs, bud, True)

# Full Training
mod_t_val_valacc, mod_t_val_tstacc, mod_t_val_valloss, mod_t_val_tstloss, mod_tay_fval_substrn_losses, mod_tay_fval_fulltrn_losses, mod_tay_fval_val_losses, mod_subset_idxs = train_model_mod_taylor(
    start_idxs, bud, True)

plot_start_epoch = 0
###### Subset Trn loss with val = VAL #############
plt.figure()
plt.plot(np.arange(plot_start_epoch, num_epochs), tay_fval_substrn_losses[plot_start_epoch:], 'b-',
         label='GLISTER-ONLINE')
plt.plot(np.arange(plot_start_epoch, num_epochs), craig_substrn_losses[plot_start_epoch:], '#750D86',
         label='CRAIG')
# plt.plot(np.arange(plot_start_epoch, num_epochs), e_craig_substrn_losses[plot_start_epoch:], 'm',
#         label='CRAIG ev epo')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_substrn_losses[plot_start_epoch:], 'g-', label='random')
if feature == 'classimb':
    plt.plot(np.arange(plot_start_epoch, num_epochs), rand_prior_substrn_losses[plot_start_epoch:], '#DD4477',
             label='random_prior')
# plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_substrn_losses[plot_start_epoch:], 'pink', label='FacLoc')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_tay_fval_substrn_losses[plot_start_epoch:], 'k-',
         label='rand_reg_GLISTER-ONLINE')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_tay_fval_substrn_losses[plot_start_epoch:], 'y',
         label='facloc_reg_GLISTER-ONLINE')
plt.plot(np.arange(plot_start_epoch, num_epochs), mod_tay_fval_substrn_losses[plot_start_epoch:], 'pink',
         label='Full Training')

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

plt.plot(np.arange(plot_start_epoch, num_epochs), tay_fval_fulltrn_losses[plot_start_epoch:], 'b-',
         label='GLISTER-ONLINE')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_fulltrn_losses[plot_start_epoch:], 'g-', label='random')
if feature == 'classimb':
    plt.plot(np.arange(plot_start_epoch, num_epochs), rand_prior_fulltrn_losses[plot_start_epoch:], '#DD4477',
             label='random_prior')
plt.plot(np.arange(plot_start_epoch, num_epochs), craig_fulltrn_losses[plot_start_epoch:], '#750D86', label='CRAIG')
# plt.plot(np.arange(plot_start_epoch, num_epochs), e_craig_fulltrn_losses[plot_start_epoch:], 'm', label='CRAIG ev epo')
# plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_fulltrn_losses[plot_start_epoch:], 'pink', label='FacLoc')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_tay_fval_fulltrn_losses[plot_start_epoch:], 'k-',
         label='rand_reg_GLISTER-ONLINE')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_tay_fval_fulltrn_losses[plot_start_epoch:], 'y',
         label='facloc_reg_GLISTER-ONLINE')
plt.plot(np.arange(plot_start_epoch, num_epochs), mod_tay_fval_fulltrn_losses[plot_start_epoch:], 'pink',
         label='Full Training')

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
plt.plot(np.arange(plot_start_epoch, num_epochs), tay_fval_val_losses[plot_start_epoch:], 'b-', label='GLISTER-ONLINE')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_val_losses[plot_start_epoch:], 'g-', label='random')
if feature == 'classimb':
    plt.plot(np.arange(plot_start_epoch, num_epochs), rand_prior_val_losses[plot_start_epoch:], '#DD4477',
             label='random_prior')
plt.plot(np.arange(plot_start_epoch, num_epochs), craig_val_losses[plot_start_epoch:], '#750D86', label='CRAIG')
# plt.plot(np.arange(plot_start_epoch, num_epochs), craig_val_losses[plot_start_epoch:], 'm', label='CRAIG ev epo')

# plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_val_losses[plot_start_epoch:], 'pink', label='FacLoc')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_tay_fval_val_losses[plot_start_epoch:], 'k-',
         label='rand_reg_GLISTER-ONLINE')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_tay_fval_val_losses[plot_start_epoch:], 'y',
         label='facloc_reg_GLISTER-ONLINE')
# DD4477
plt.plot(np.arange(plot_start_epoch, num_epochs), mod_tay_fval_val_losses[plot_start_epoch:], 'pink',
         label='Full Training')

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
# print('*| Facility Location             |', fv1, '  | ', ft1, ' |', file=logfile)
print('*| Taylor with Validation=VAL     |', t_val_valacc, '  | ', t_val_tstacc, ' |', file=logfile)
print('*| Random Selection               |', rv1, '  | ', rt1, ' |', file=logfile)
if feature == 'classimb':
    print('*| Random Prior Selection               |', rpv1, '  | ', rpt1, ' |', file=logfile)
# print('*| CRAIG Selected every epoch               |', e_craig_valacc, '  | ', e_craig_tstacc, ' |', file=logfile)
print('*| CRAIG Selection               |', craig_valacc, '  | ', craig_tstacc, ' |', file=logfile)
print('*| Full Training               |', mod_t_val_valacc, '  | ', mod_t_val_tstacc, ' |', file=logfile)
print('*| random regularized Taylor after training               |', rand_t_val_valacc, '  | ', rand_t_val_tstacc, ' |',
      file=logfile)
print('*| facloc regularizec Taylor after training               |', facloc_reg_t_val_valacc, '  | ',
      facloc_reg_t_val_tstacc, ' |', file=logfile)
print("\n", file=logfile)

print("=========Random Results==============", file=logfile)
print("*Rand Validation LOSS:", rv2, file=logfile)
print("*Rand Test Data LOSS:", rt2, file=logfile)
print("*Rand Full Trn Data LOSS:", rand_fulltrn_losses[-1], file=logfile)

print("=========CRAIG Results==============", file=logfile)
print("*CRAIG Validation LOSS:", craig_valloss, file=logfile)
print("*CRAIG Test Data LOSS:", craig_tstloss, file=logfile)
print("*CRAIG Full Trn Data LOSS:", craig_fulltrn_losses[-1], file=logfile)

'''print("=========CRAIG Selected every epoch Results==============", file=logfile)
print("*CRAIG Validation LOSS:", e_craig_valloss, file=logfile)
print("*CRAIG Test Data LOSS:", e_craig_tstloss, file=logfile)
print("*CRAIG Full Trn Data LOSS:", e_craig_fulltrn_losses[-1], file=logfile)'''

'''print("=========FacLoc Results==============", file=logfile)
print("*Facloc Validation LOSS:", fv2, file=logfile)
print("*Facloc Test Data LOSS:", ft2, file=logfile)
print("*Facloc Full Trn Data LOSS:", facloc_fulltrn_losses[-1], file=logfile)'''

print("=========Online Selection Taylor with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", t_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", t_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", tay_fval_fulltrn_losses[-1], file=logfile)

print("=========Random Regularized Online Selection Taylor with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", rand_t_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", rand_t_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", rand_tay_fval_fulltrn_losses[-1], file=logfile)

print("=========Facility Location Loss regularized Online Selection Taylor with Validation Set===================",
      file=logfile)
print("*Taylor v=VAL Validation LOSS:", facloc_reg_t_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", facloc_reg_t_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", facloc_reg_tay_fval_fulltrn_losses[-1], file=logfile)
print("=============================================================================================", file=logfile)
print("---------------------------------------------------------------------------------------------", file=logfile)
print("\n", file=logfile)

subset_idxs = list(subset_idxs)
with open(all_logs_dir + '/one_step_subset_selected.txt', 'w') as log_file:
    print(subset_idxs, file=log_file)

rand_subset_idxs = list(rand_subset_idxs)
with open(all_logs_dir + '/rand_reg_one_step_subset_selected.txt', 'w') as log_file:
    print(rand_subset_idxs, file=log_file)

rand_prior_idxs = list(rand_prior_idxs)
with open(all_logs_dir + '/rand_prior_subset_selected.txt', 'w') as log_file:
    print(rand_prior_idxs, file=log_file)

craig_subset_idxs = list(craig_subset_idxs)
with open(all_logs_dir + '/craig_subset_selected.txt', 'w') as log_file:
    print(craig_subset_idxs, file=log_file)

facloc_reg_subset_idxs = list(facloc_reg_subset_idxs)
with open(all_logs_dir + '/facloc_reg_one_step_subset_selected.txt', 'w') as log_file:
    print(facloc_reg_subset_idxs, file=log_file)

random_subset_idx = list(random_subset_idx)
with open(all_logs_dir + '/random_subset_selected.txt', 'w') as log_file:
    print(random_subset_idx, file=log_file)

facloc_idxs = list(facloc_idxs)
with open(all_logs_dir + '/facloc_subset_selected.txt', 'w') as log_file:
    print(facloc_idxs, file=log_file)