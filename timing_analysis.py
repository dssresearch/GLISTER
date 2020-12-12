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
from models.set_function_all import SetFunctionFacLoc, \
    SetFunctionBatch  # as SetFunction #SetFunctionCompare
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler
from models.set_function_grad_computation_taylor import Small_GlisterSetFunction_Closed as SetFunctionTaylor
#from models.set_function_craig import SetFunction2 as CRAIG
from models.set_function_craig import SetFunctionCRAIG_Super as CRAIG
from models.set_function_ideas import SetFunctionTaylorDeep_ReLoss_Mean
from sklearn.model_selection import train_test_split
from utils.custom_dataset import load_dataset_custom, load_mnist_cifar
import math
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score

device = "cpu" #if torch.cuda.is_available() else "cpu"
# #device = "cpu"
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

all_logs_dir = './results/Timing/' + data_name + '/' + str(fraction) + '/' + str(select_every)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt') 

logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
           '_selEvery:' + str(select_every) + '_variant' + str(warm_method) + '_runs' + str(num_runs)
print(exp_name)
#print("=======================================", file=logfile)
#print(exp_name, str(exp_start_time), file=logfile)

if data_name in ['dna','sklearn-digits','satimage','svmguide1','letter','shuttle','ijcnn1','sensorless','connect_4','sensit_seismic','usps']:
    fullset, valset, testset, data_dims, num_cls = load_dataset_custom(datadir, data_name,feature=feature, isnumpy=True)
elif data_name in ['mnist' , "fashion-mnist"]:
    fullset, valset, testset, num_cls = load_mnist_cifar(datadir, data_name,feature=feature)


'''if data_name == 'mnist' or data_name == "fashion-mnist":
    x_trn, y_trn = fullset.data, fullset.targets
    x_tst, y_tst = testset.data, testset.targets
    x_trn = x_trn.view(x_trn.shape[0], -1)
    x_tst = x_tst.view(x_tst.shape[0], -1)
    # Get validation data: Its 10% of the entire (full) training data
    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
else:'''
x_trn, y_trn = torch.from_numpy(fullset[0]).float(), torch.from_numpy(fullset[1]).long()
x_tst, y_tst = torch.from_numpy(testset[0]).float(), torch.from_numpy(testset[1]).long()
x_val, y_val = torch.from_numpy(valset[0]).float(), torch.from_numpy(valset[1]).long()

print('-----------------------------------------')
#print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
# print(y_trn.shape, y_val.shape, y_tst.shape)

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


def train_model_craig(start_rand_idxs, bud, convex=True,every=False):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
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
   
    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    #dataset = TensorDataset(x_trn, y_trn)
    #dataloader = DataLoader(dataset, shuffle=False, batch_size=train_batch_size)
    #setf = CRAIG(device, train_loader, True)
    setf = CRAIG(device, x_trn,y_trn, True)
    # idxs = start_rand_idxs
    idxs, gammas = setf.class_wise(bud, model)
    #gammas = gammas.type(torch.float)/N

    exp_start_time_craig = time.time()
    for i in range(num_epochs):
        #print(i)
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        optimizer.zero_grad()
        scores = model(inputs)
        losses = criterion_nored(scores, targets)
        loss = torch.dot(torch.from_numpy(np.array(gammas)).to(device).type(torch.float)/N, losses)
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

        timing[i] = time.time()
        val_acc[i] = val_accu
        tst_acc[i] = tst_accu
        if not convex :
            if not every and (i + 1) % select_every == 0:
                idxs, gammas = setf.class_wise(bud, model)
                #gammas = gammas.type(torch.float)/N
            else:
                idxs, gammas = setf.class_wise(bud, model)
                #gammas = gammas.type(torch.float)/N
    
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_accu)
    print("Test Data Loss and Accuracy:", tst_loss.item(), tst_accu)
    print('-----------------------------------')
    return val_acc, tst_acc, timing - exp_start_time_craig


def train_model_taylor(func_name, start_rand_idxs=None, bud=None, valid=True, fac_loc_idx=None):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = ThreeLayerNet(M, num_cls, 100, 100)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model = model.to(device)

    if func_name == 'Stepwise':
        step_bud = math.ceil(bud * select_every / num_epochs)
        remainList = set(list(range(N)))

        idxs = list(np.random.choice(N, size=step_bud, replace=False))
        remainList = remainList.difference(idxs)

    else:
        idxs = start_rand_idxs

    if func_name == 'Facloc Regularized':
        x_val1 = torch.cat([x_val, x_trn[fac_loc_idx]], dim=0)
        y_val1 = torch.cat([y_val, y_trn[fac_loc_idx]], dim=0)

    total_idxs = list(np.arange(len(y_trn)))

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if func_name == 'Full OneStep':
        setf_model = SetFunctionBatch(x_trn, y_trn, x_val, y_val, valid, model,
                                      criterion, criterion_nored, learning_rate, device)

    elif func_name == "Taylor on Logit":
        setf_model = SetFunctionTaylorDeep_ReLoss_Mean(x_trn, y_trn, train_batch_size_for_greedy, x_val, y_val, valid,
                                                       model,
                                                       criterion, criterion_nored, learning_rate, device, N)

    elif func_name == 'Facility Location':
        setf_model = SetFunctionFacLoc(device, train_loader_greedy)
        idxs = setf_model.lazy_greedy_max(bud, model)

    elif func_name == 'Facloc Regularized':
        setf_model = SetFunctionTaylor(x_trn, y_trn, x_val1, y_val1, model,
                                       criterion, criterion_nored, learning_rate, device, num_cls)

    else:
        setf_model = SetFunctionTaylor(x_trn, y_trn, x_val, y_val, model,
                                       criterion, criterion_nored, learning_rate, device, num_cls)

    if func_name == 'Taylor Online':
        print("Starting Online OneStep Run with taylor on loss!")
    elif func_name == 'Full OneStep':
        print("Starting Online OneStep Run without taylor!")
    elif func_name == 'Taylor on Logit':
        print("Starting Online OneStep Run with taylor on logit!")
    elif func_name == 'Stepwise':
        print("Starting Online Stepwise OneStep Run with taylor!")
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
    elif func_name == 'Random Perturbation':
        print("Starting Online OneStep Run with taylor with random perturbation!")
    elif func_name == 'Proximal':
        print("Starting Online Proximal OneStep Run with taylor!")
    
    timing = np.zeros(num_epochs)
    substrn_grads = []
    tst_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    # idxs = start_rand_idxs
    
    exp_start_time_onestep = time.time()
    
    for i in range(num_epochs):
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
        temp1 = torch.autograd.grad(loss, model.parameters())
        grad_value = torch.norm(torch.cat((temp1[0], temp1[1].view(-1, 1)) ,dim=1).flatten()).item()
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
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

        timing[i] = time.time()
        val_acc[i] = val_accu
        tst_acc[i] = tst_accu

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if ((i + 1) % select_every == 0) and func_name not in ['Facility Location', 'Random',"KNNSB"]:
            substrn_grads.append(grad_value)
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            t_ng_start = time.time()
            if func_name == 'Stepwise':
                new_idxs = setf_model.naive_greedy_max(step_bud, clone_dict, list(remainList))  # , grads_idxs
                remainList = remainList.difference(new_idxs)
                idxs.extend(new_idxs)

            elif func_name == 'Random Greedy':
                new_idxs = setf_model.naive_greedy_max(int(0.9 * bud), clone_dict)
                rem_idxs = list(set(total_idxs).difference(set(new_idxs)))
                new_idxs.extend(list(np.random.choice(rem_idxs, size=int(0.1 * bud), replace=False)))
                idxs = new_idxs

            elif func_name == 'Online Random':
                state = np.random.get_state()
                np.random.seed(i)
                idxs = np.random.choice(N, size=bud, replace=False)
                np.random.set_state(state)

            elif func_name == 'Random Perturbation':
                new_idxs = setf_model.naive_greedy_max(bud, clone_dict, None, None, True)
                idxs = new_idxs

            elif func_name == 'Proximal':
                previous = torch.zeros(N, device=device)
                previous[idxs] = 1.0
                new_idxs = setf_model.naive_greedy_max(bud, clone_dict, None, previous)
                idxs = new_idxs

            else:
                new_idxs = setf_model.naive_greedy_max(bud, clone_dict)  # , grads_idxs
                idxs = new_idxs  # update the current set
            model.load_state_dict(cached_state_dict)

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_accu)
    print("Test Data Loss and Accuracy:", tst_loss.item(), tst_accu)
    print('-----------------------------------')
    return val_acc, tst_acc, timing - exp_start_time_onestep


def train_model_mod_taylor(start_rand_idxs, bud, valid):
    torch.manual_seed(42)
    np.random.seed(42)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # model = ThreeLayerNet(M, num_cls, 5, 5)
    if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting Full Training!")
    
    timing = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    # idxs = start_rand_idxs
    exp_start_time_full = time.time()
    batch_wise_indices = list(BatchSampler(RandomSampler(x_trn), int(bud), drop_last=False))
    for i in range(num_epochs):
        for batch_idx in batch_wise_indices:
            inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
            #inputs, targets = x_trn[batch_idx], y_trn[batch_idx]
            optimizer.zero_grad()
            scores = model(inputs)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()

        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn, y_trn
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)
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

        timing[i] = time.time()
        val_acc[i] = val_accu
        tst_acc[i] = tst_accu


    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_accu)
    print("Test Data Loss and Accuracy:", tst_loss.item(), tst_accu)
    print('-----------------------------------')

    return val_acc, tst_acc, timing - exp_start_time_full

start_idxs = np.random.choice(N, size=bud, replace=False)
random_subset_idx = [x for x in start_idxs]
## KnnSB selection with Flag = TRN and FLAG = VAL
#knn_idxs_flag_trn = perform_knnsb_selection(datadir, data_name, fraction, selUsing='trn')
#knn_idxs_flag_val = perform_knnsb_selection(datadir, data_name, fraction, selUsing='val')

# CRAIG Run
#print(time.time())
#craig_valacc, craig_tstacc, craig_timing = train_model_craig(start_idxs, bud, False, False)
#print(time.time())

# Every epoch CRAIG Run
#e_craig_valacc, e_craig_tstacc, e_craig_valloss, e_craig_tstloss, e_craig_substrn_losses, e_craig_fulltrn_losses, \
#e_craig_val_losses, e_craig_subset_idxs = train_model_craig(start_idxs, bud, False,True)


# Random Run
#rv1, rt1, rv2, rt2, rand_substrn_losses, rand_fulltrn_losses, rand_val_losses, idxs, rand_grads = train_model_taylor('Random',start_idxs)

# Facility Location Run
#fv1, ft1, fv2, ft2, facloc_substrn_losses, facloc_fulltrn_losses, facloc_val_losses, facloc_idxs, facloc_grads = train_model_taylor(
#    'Facility Location', None, bud)

## Training with KnnSB idxs with Flag = TRN and FLAG = VAL
#knn_valacc_flagtrn, knn_tstacc_flagtrn, knn_valloss_flagtrn, knn_tstloss_flagtrn, knn_ftrn_substrn_losses, knn_ftrn_fulltrn_losses,\
# knn_ftrn_val_losses, knn_ftrn_idxs, knn_ftrn_grads = train_model_taylor("KNNSB",knn_idxs_flag_trn)

#knn_valacc_flagval, knn_tstacc_flagval, knn_valloss_flagval, knn_tstloss_flagval, knn_fval_substrn_losses, knn_fval_fulltrn_losses,\
# knn_fval_val_losses, knn_fval_idxs, knn_fval_grads = train_model_taylor("KNNSB",knn_idxs_flag_val)


# Online algo run
print(time.time())
t_val_valacc, t_val_tstacc, t_val_timing = train_model_taylor('Taylor Online', start_idxs, bud, True)
print(time.time())

# Facility Location OneStep Runs
#facloc_reg_t_val_valacc, facloc_reg_t_val_tstacc, facloc_reg_t_val_valloss, facloc_reg_t_val_tstloss, facloc_reg_tay_fval_substrn_losses, facloc_reg_tay_fval_fulltrn_losses, facloc_reg_tay_fval_val_losses, facloc_reg_subset_idxs, facloc_reg_grads = train_model_taylor(
#    'Facloc Regularized', start_idxs, bud, True, facloc_idxs)

# Randomized Greedy Taylor OneStep Runs
#rand_t_val_valacc, rand_t_val_tstacc, rand_t_val_valloss, rand_t_val_tstloss, rand_tay_fval_substrn_losses, rand_tay_fval_fulltrn_losses, rand_tay_fval_val_losses, rand_subset_idxs, rand_reg_grads = train_model_taylor(
#    'Random Greedy', start_idxs, bud, True)

# Taylor After Training
print(time.time())
mod_t_valacc, mod_t_tstacc, mod_t_timing = train_model_mod_taylor(start_idxs, bud, True)
print(time.time())

#craig_timing = craig_timing- exp_start_time_craig
#mod_t_timing = mod_t_timing  - exp_start_time_full
#t_val_timing = t_val_timing  - exp_start_time_onestep

###### Test accuray #############

plt.figure()
#plt.plot(craig_timing, craig_tstacc,'g-' , label='CRAIG')
plt.plot(mod_t_timing, mod_t_tstacc, 'orange', label='full training')
plt.plot(t_val_timing, t_val_tstacc, 'b-', label='GLISTER')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Test accuracy')
plt.title('Test Accuracy vs Time ' + data_name + '_' + str(fraction))
plt_file = path_logfile + '_' + str(fraction) + 'tst_accuracy_v=VAL.png'
plt.savefig(plt_file)
plt.clf()

########################################################################
###### Validation #############

plt.figure()
#plt.plot(craig_timing, craig_valacc,'g-' , label='CRAIG')
plt.plot(mod_t_timing, mod_t_valacc, 'orange', label='full training')
plt.plot(t_val_timing, t_val_valacc, 'b-', label='GLISTER')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Validation accuracy')
plt.title('Validation Accuracy vs Time ' + data_name + '_' + str(fraction) )
plt_file = path_logfile + '_' + str(fraction) + 'val_accuracy_v=VAL.png'
plt.savefig(plt_file)
plt.clf()

"""
print("CRAIG",file=logfile)
print('---------------------------------------------------------------------',file=logfile)


val = "Validation Accuracy,"
tst = "Test Accuracy,"
timing = "Time,"

for i in range(num_epochs):
    timing = timing+"," +str(craig_timing[i])
    val = val+"," +str(craig_valacc[i])
    tst = tst+"," +str(craig_tstacc[i])
    
print(time,file=logfile)
print(val,file=logfile)
print(tst,file=logfile)
"""
print("GLISTER",file=logfile)
print('---------------------------------------------------------------------',file=logfile)


val = "Validation Accuracy,"
tst = "Test Accuracy,"
timing = "Time,"

for i in range(num_epochs):
    timing = timing+"," +str(t_val_timing[i])
    val = val+"," +str(t_val_valacc[i])
    tst = tst+"," +str(t_val_tstacc[i])
    
print(timing,file=logfile)
print(val,file=logfile)
print(tst,file=logfile)

print("Full Training set",file=logfile)
print('---------------------------------------------------------------------',file=logfile)


val = "Validation Accuracy,"
tst = "Test Accuracy,"
timing = "Time,"

for i in range(num_epochs):
    timing = timing+"," +str(mod_t_timing[i])
    val = val+"," +str(mod_t_valacc[i])
    tst = tst+"," +str(mod_t_tstacc[i])
    
print(timing,file=logfile)
print(val,file=logfile)
print(tst,file=logfile)

logfile.close()