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
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
from models.set_function_all import SetFunctionFacLoc
from torch.utils.data.sampler import SubsetRandomSampler
# from models.simpleNN_net import ThreeLayerNet
#from models.set_function_craig import DeepSetFunction as CRAIG
from models.cifar_set_function_act_learn import SetFunctionLoader_2 as SetFunction
from models.cifar_set_function_act_learn import WeightedSetFunctionLoader as WtSetFunction
import math
from models.resnet import ResNet18
from utils.data_utils import load_dataset_pytorch
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler
from torch.distributions import Categorical

import sys
sys.path.append('./badge/')
from badge.my_run import return_accuracies

def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss


def write_knndata(datadir,dset_name,x_val, y_val):#, x_trn, y_trn,  x_tst, y_tst, dset_name):
    ## Create VAL data
    subprocess.run(["mkdir", "-p", datadir])
    trn_filepath = os.path.join(datadir, 'knn_' + dset_name + '.trn')
    val_filepath = os.path.join(datadir, 'knn_' + dset_name + '.val')
    tst_filepath = os.path.join(datadir, 'knn_' + dset_name + '.tst')

    if os.path.exists(val_filepath): #and os.path.exists(tst_filepath):
        return
    #os.path.exists(trn_filepath) and 
    # x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
    #trndata = np.c_[x_trn.cpu().numpy(), y_trn.cpu().numpy()]
    valdata = np.c_[x_val.cpu().numpy(), y_val.cpu().numpy()]
    #tstdata = np.c_[x_tst.cpu().numpy(), y_tst.cpu().numpy()]
    # Write out the trndata
    
    #np.savetxt(trn_filepath, trndata, fmt='%.6f')
    np.savetxt(val_filepath, valdata, fmt='%.6f')
    #np.savetxt(tst_filepath, tstdata, fmt='%.6f')
    return

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:0"
print("Using Device:", device)

## Convert to this argparse
datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
no_select = int(sys.argv[5])
warm_method = sys.argv[6]
num_runs = sys.argv[7]
print_every = 50

learning_rate = 0.05
all_logs_dir = './results/ActLearn/' + data_name  + '_grad/' + str(fraction) + '/' + str(no_select)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
           '_selEvery:' + str(no_select) + '_variant' + str(warm_method) + '_runs' + str(num_runs)
print(exp_name)
exp_start_time = datetime.datetime.now()
print("=======================================", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)
if data_name == 'cifar10':
    fullset, valset, testset, num_cls = load_dataset_pytorch(datadir, data_name)
    # Validation Data set is 10% of the Entire Trainset.
    validation_set_fraction = 0.1
    num_fulltrn = len(fullset)
    num_val = int(num_fulltrn * validation_set_fraction)
    num_trn = num_fulltrn - num_val
    trainset, validset = random_split(fullset, [num_trn, num_val])#,generator=torch.Generator().manual_seed(42))
    trn_batch_size = 128
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
            #x_tst_new = torch.cat([x_tst_new, inputs.view(tst_batch_size, -1)], dim=0)

write_knndata(datadir, data_name,x_val_new, y_val)#x_trn, y_trn, , x_tst_new, y_tst, data_name)
print('-----------------------------------------')
print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
# print(y_trn.shape, y_val.shape, y_tst.shape)

def perform_knnsb_selection(datadir, dset_name,remain, budget, selUsing):


    trndata = np.c_[x_trn.cpu()[remain].view(len(remain), -1), y_trn.cpu()[remain]]
    # Write out the trndata
    trn_filepath = os.path.join(datadir, 'act_'+'_knn_' + dset_name + '.trn')
    np.savetxt(trn_filepath, trndata, fmt='%.6f')
    
    run_path = './run_data/'
    output_dir = run_path + 'KNNSubmod_' + dset_name + '/'
    indices_file = output_dir + 'act_'+'_KNNSubmod_' + str((int)(budget*100)) + ".subset"

    if selUsing == 'val':
        val_filepath = os.path.join(datadir, 'knn_' + dset_name + '.val')
    else:
        val_filepath = trn_filepath

    subprocess.call(["mkdir","-p", output_dir])
    knnsb_args = []
    knnsb_args.append('../build/KNNSubmod')
    knnsb_args.append(trn_filepath)
    knnsb_args.append(val_filepath)
    knnsb_args.append(" ")  # File delimiter!!
    knnsb_args.append(str(no_points/x_trn[remain].shape[0]))
    knnsb_args.append(indices_file)
    knnsb_args.append("1")  # indicates cts data. Deprecated.
    print("Obtaining the subset")
    #print(knnsb_args)
    subprocess.run(knnsb_args)
    print("finished selection")
    # Can make it return the indices_file if using with other function. 
    idxs_knnsb = np.genfromtxt(indices_file, delimiter=',', dtype=int) # since they are indices!
    return idxs_knnsb

N, M = x_trn.shape
n_val = x_val_new.shape[0]
#bud = int(fraction * N)
no_points = math.ceil(fraction*N/no_select)
print("Budget, fraction and N:", no_points, fraction, N)
# Transfer all the data to GPU
d_t = time.time()
x_trn, y_trn = x_trn.to('cpu'), y_trn.to('cpu')
x_val, y_val = x_val.to('cpu'), y_val.to('cpu')
print("Transferred data to device in time:", time.time() - d_t)
print_every = 3

def run_stochastic_Facloc(data, targets, budget):
    model = ResNet18(num_cls)
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

def active_learning_taylor(func_name,start_rand_idxs=None, bud=None, valid=True,fac_loc_idx=None):
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = ResNet18(num_cls)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    remainList = set([i for i in range(N)])
    idxs = list(idxs)
    remainList = remainList.difference(idxs)

    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),pin_memory=True)

    if func_name == 'Taylor Online':
        print("Starting Online OneStep Run with taylor on loss!")
    elif func_name == 'Full OneStep':
        print("Starting Online OneStep Run without taylor!")
    elif func_name == 'Facloc Regularized':
        print("Starting Facility Location Regularized Online OneStep Run with taylor!")
    elif func_name == 'Random Greedy':
        print("Starting Randomized Greedy Online OneStep Run with taylor!")
    elif func_name == 'Facility Location':
         print("Starting Facility Location!")
    elif func_name == 'Random':
        print("Starting Random Run!")
    elif func_name == 'Random Perturbation':
        print("Starting Online OneStep Run with taylor with random perturbation!")
    elif func_name == "FASS":
        print("Filtered Active Submodular Selection(FASS)!")
    #elif func_name == 'Proximal':
        #print("Starting Online Proximal OneStep Run with taylor!")
    #elif func_name == 'Taylor on Logit':
    #    print("Starting Online OneStep Run with taylor on logit!")
    
    
    # if valid:
    #     print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    # else:
    #     print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)

    val_accies = np.zeros(no_select)
    test_accies = np.zeros(no_select)
    unlab_accies = np.zeros(no_select)
    # idxs = start_rand_idxs

    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    fn = nn.Softmax(dim=1)
    for n in range(no_select):

        model.train()
        for i in range(num_epochs):

            accFinal = 0.
            for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                # targets can have non_blocking=True.
                x, y = inputs.to(device), targets.to(device, non_blocking=True)
                #x, y = Variable(x.cuda()), Variable(y.cuda())
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
                loss.backward()

                if (i % 50 == 0) and (accFinal < 0.2): # reset if not converging
                    model =  model.apply(weight_reset).cuda()
                    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

                # clamp gradients, just in case
                for p in filter(lambda p: p.grad is not None, model.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

                optimizer.step()

        print( n+1,'Time', 'SubsetTrn', loss.item())#, ,FullTrn,ValLoss: full_trn_loss.item(), val_loss.item())

        curr_X_trn = x_trn[list(remainList)]
        #curr_Y_trn = y_trn[list(remainList)]

        model.eval()
        with torch.no_grad():
            '''full_trn_out = model(x_trn)
            full_trn_loss = criterion(full_trn_out, y_trn).mean()
            sub_trn_out = model(x_trn[idxs])
            sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()'''

            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, val_predict = outputs.max(1)
                correct += val_predict.eq(targets).sum().item()
                total += targets.size(0)
            val_acc = 100 * correct / total

            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, tst_predict = outputs.max(1)
                correct += tst_predict.eq(targets).sum().item()
                total += targets.size(0)
            tst_acc = 100.0 * correct / total
            
            remloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(list(remainList)),
                                                   pin_memory=True)

            correct = 0
            total = 0
            cnt = 0
            predictions = []
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                predictions.append(outputs)
                _, rem_predict = outputs.max(1)
                if cnt == 0:
                    y_rem_trn = rem_predict
                    cnt = cnt + 1
                else:
                    y_rem_trn = torch.cat([y_rem_trn, rem_predict], dim=0)
                
                correct += rem_predict.eq(targets).sum().item()
                total += targets.size(0)
            rem_acc = 100 * correct / total

        val_accies[n] = val_acc
        test_accies[n] = tst_acc
        unlab_accies[n] = rem_acc

        #if ((i + 1) % select_every == 0) and func_name not in ['Facility Location','Random']:
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
        cached_state_dict = copy.deepcopy(model.state_dict())
        clone_dict = copy.deepcopy(model.state_dict())
        # Dont put the logs for Selection on logfile!!
        # print("With Taylor approximation",file=logfile)
        # print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
        #t_ng_start = time.time()

        if func_name == 'Random Greedy':
            new_idxs = setf_model.naive_greedy_max(curr_X_trn,y_rem_trn,int(0.9 * no_points), clone_dict)
            new_idxs = list(np.array(list(remainList))[new_idxs])
            
            remainList = remainList.difference(new_idxs)
            new_idxs.extend(list(np.random.choice(list(remainList), size=int(0.1 * no_points), replace=False)))
            remainList = remainList.difference(new_idxs)
            idxs.extend(new_idxs)

        elif func_name == "FASS":

            cnt = 0
            for pre in predictions:
                soft = fn(pre)
                if cnt == 0:
                    entropy2 = Categorical(probs = soft).entropy()
                    cnt = cnt + 1
                else:
                    entropy2 = torch.cat([entropy2, Categorical(probs = soft).entropy()], dim=0)

            #print(entropy2.shape)
            if 5*no_points < entropy2.shape[0]:
                values,indices = entropy2.topk(5*no_points)
                indices = list(np.array(list(remainList))[indices.cpu()])
            else:
                indices = list(remainList)

            knn_idxs_flag_val = perform_knnsb_selection(datadir, data_name, indices, fraction, selUsing='val') 
            
            ##print(len(knn_idxs_flag_val),len(indices))
            knn_idxs_flag_val = list(np.array(indices)[knn_idxs_flag_val])

            remainList = remainList.difference(knn_idxs_flag_val)
            idxs.extend(knn_idxs_flag_val)

        elif func_name == 'Random':
            state = np.random.get_state()
            np.random.seed(n*n)
            new_idxs = np.random.choice(list(remainList), size=no_points, replace=False)
            np.random.set_state(state)
            remainList = remainList.difference(new_idxs)
            idxs.extend(new_idxs)

            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs),
                                                           pin_memory=True)

        elif func_name == 'Facility Location':

            new_idxs = run_stochastic_Facloc(curr_X_trn, y_rem_trn, no_points)
            new_idxs = np.array(list(remainList))[new_idxs]

            remainList = remainList.difference(new_idxs)
            idxs.extend(new_idxs)

            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs),
                                                           pin_memory=True)

        else: 
            new_idxs = setf_model.naive_greedy_max(curr_X_trn,rem_predict,no_points, clone_dict)  # , grads_idxs
            new_idxs = np.array(list(remainList))[new_idxs]

            remainList = remainList.difference(new_idxs)
            idxs.extend(new_idxs) 

        '''elif func_name == 'Proximal':
            previous = torch.zeros(N,device=device)
            previous[idxs] = 1.0 
            new_idxs = setf_model.naive_greedy_max(bud, clone_dict,None,previous)
            idxs = new_idxs'''

        # print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
        # print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
        model.load_state_dict(cached_state_dict)

    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    
    return val_accies, test_accies, unlab_accies, idxs

def act_learn_online(function,start_rand_idxs, bud,lam=0.9):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = ResNet18(num_cls)
    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = SetFunction(trainset, x_val, y_val, model, criterion,
                             criterion_nored, learning_rate, device, num_cls, 1000)
    print("Starting Greedy Online OneStep Run with taylor!")
    remainList = set([i for i in range(N)])
    idxs = list(idxs)
    remainList = remainList.difference(idxs)

    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),pin_memory=True)

    val_accies = np.zeros(no_select)
    test_accies = np.zeros(no_select)
    unlab_accies = np.zeros(no_select)

    for n in range(no_select):
        model.train()
        for i in range(num_epochs):
            actual_idxs = np.array(trainset.indices)[idxs]
            batch_wise_indices = [actual_idxs[x] for x in list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))]
            subtrn_loss = 0
            for batch_idx in batch_wise_indices:
                inputs = torch.cat(
                    [fullset[x][0].view(-1, 3, fullset[x][0].shape[1], fullset[x][0].shape[2]) for x in batch_idx],
                    dim=0).type(torch.float)
                targets = torch.tensor([fullset[x][1] for x in batch_idx])
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True) # targets can have non_blocking=True.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                for p in filter(lambda p: p.grad is not None, model.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
                optimizer.step()

        curr_X_trn = x_trn[list(remainList)]
        #curr_Y_trn = y_trn[list(remainList)]

        model.eval()
        with torch.no_grad():
            '''full_trn_out = model(x_trn)
            full_trn_loss = criterion(full_trn_out, y_trn).mean()
            sub_trn_out = model(x_trn[idxs])
            sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()'''

            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, val_predict = outputs.max(1)
                correct += val_predict.eq(targets).sum().item()
                total += targets.size(0)
            val_acc = 100 * correct / total

            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, tst_predict = outputs.max(1)
                correct += tst_predict.eq(targets).sum().item()
                total += targets.size(0)
            tst_acc = 100.0 * correct / total
            
            remloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(list(remainList)),
                                                   pin_memory=True)

            correct = 0
            total = 0
            cnt = 0
            predictions = []
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                predictions.append(outputs)
                _, rem_predict = outputs.max(1)
                if cnt == 0:
                    y_rem_trn = rem_predict
                    cnt = cnt + 1
                else:
                    y_rem_trn = torch.cat([y_trn, rem_predict], dim=0)
                
                correct += rem_predict.eq(targets).sum().item()
                total += targets.size(0)
            rem_acc = 100 * rem_correct / rem_total

        val_accies[n] = val_acc
        test_accies[n] = tst_acc
        unlab_accies[n] = rem_acc

        print('Epoch:', i + 1, 'test Accuracy', tst_acc)
            
        cached_state_dict = copy.deepcopy(model.state_dict())
        clone_dict = copy.deepcopy(model.state_dict())
        #prev_idxs = idxs
        print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
        if function == 'Random Greedy':
            new_idxs, grads_idxs = setf_model.naive_greedy_max(int(lam * bud),list(remainList), clone_dict)
            new_idxs = list(np.array(list(remainList))[new_idxs])
            rem_idxs = list(remainList.difference(new_idxs))
            new_idxs.extend(list(np.random.choice(rem_idxs, size=int((1 - lam) * bud), replace=False)))
        else:
            new_idxs, grads_idxs = setf_model.naive_greedy_max(bud,list(remainList), clone_dict)
            new_idxs = list(np.array(list(remainList))[new_idxs])
        remainList = remainList.difference(new_idxs)
        idxs.extend(new_idxs)
        print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
        model.load_state_dict(cached_state_dict)
        ### Change the subset_trnloader according to new found indices: subset_idxs
        subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
            #print(len(list(set(prev_idxs).difference(set(idxs)))) + len(list(set(idxs).difference(set(prev_idxs)))))
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
   
    return val_accies, test_accies, unlab_accies, idxs

def facloc_reg_train_model_online_taylor(start_rand_idxs, facloc_idxs, bud, lam):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = ResNet18(num_cls)
    val_plus_facloc_idxs = [trainset.indices[x] for x in facloc_idxs]
    val_plus_facloc_idxs.extend(validset.indices)
    cmb_set = torch.utils.data.Subset(fullset, val_plus_facloc_idxs)
    cmbloader = torch.utils.data.DataLoader(cmb_set, batch_size=1000,
                                            shuffle=False, pin_memory=True)
    for batch_idx, (inputs, targets) in enumerate(cmbloader):
        if batch_idx == 0:
            x_cmb = inputs
            y_cmb = targets
        else:
            x_cmb = torch.cat([x_cmb, inputs], dim=0)
            y_cmb = torch.cat([y_cmb, targets], dim=0)
    model = model.to(device)
    idxs = start_rand_idxs
    #total_idxs = list(np.arange(len(y_trn)))
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = WtSetFunction(trainset, x_cmb, y_cmb, len(facloc_idxs), lam, model, criterion,
                             criterion_nored, learning_rate, device, num_cls, 1000)
    print("Starting Facloc regularized Greedy Online OneStep Run with taylor!")

    remainList = set([i for i in range(N)])
    idxs = list(idxs)
    remainList = remainList.difference(idxs)

    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                   pin_memory=True)
    val_accies = np.zeros(no_select)
    test_accies = np.zeros(no_select)
    unlab_accies = np.zeros(no_select)

    for n in range(no_select):
        model.train()

        for i in range(0, num_epochs):
            actual_idxs = np.array(trainset.indices)[idxs]
            batch_wise_indices = [actual_idxs[x] for x in list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))]
            subtrn_loss = 0
            for batch_idx in batch_wise_indices:
                inputs = torch.cat(
                    [fullset[x][0].view(-1, 3, fullset[x][0].shape[1], fullset[x][0].shape[2]) for x in batch_idx],
                    dim=0).type(torch.float)
                targets = torch.tensor([fullset[x][1] for x in batch_idx])
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                subtrn_loss += loss.item()
                loss.backward()


                for p in filter(lambda p: p.grad is not None, model.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
                optimizer.step()

        curr_X_trn = x_trn[list(remainList)]
        #curr_Y_trn = y_trn[list(remainList)]

        model.eval()
        with torch.no_grad():
            '''full_trn_out = model(x_trn)
            full_trn_loss = criterion(full_trn_out, y_trn).mean()
            sub_trn_out = model(x_trn[idxs])
            sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()'''

            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, val_predict = outputs.max(1)
                correct += val_predict.eq(targets).sum().item()
                total += targets.size(0)
            val_acc = 100 * correct / total

            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, tst_predict = outputs.max(1)
                correct += tst_predict.eq(targets).sum().item()
                total += targets.size(0)
            tst_acc = 100.0 * correct / total
            
            remloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(list(remainList)),
                                                   pin_memory=True)

            correct = 0
            total = 0
            cnt = 0
            predictions = []
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                predictions.append(outputs)
                _, rem_predict = outputs.max(1)
                if cnt == 0:
                    y_rem_trn = rem_predict
                    cnt = cnt + 1
                else:
                    y_rem_trn = torch.cat([y_trn, rem_predict], dim=0)
                
                correct += rem_predict.eq(targets).sum().item()
                total += targets.size(0)
            rem_acc = 100 * rem_correct / rem_total

        val_accies[n] = val_acc
        test_accies[n] = tst_acc
        unlab_accies[n] = rem_acc

        print('Epoch:', i + 1, 'test Accuracy', tst_acc)
        
        cached_state_dict = copy.deepcopy(model.state_dict())
        clone_dict = copy.deepcopy(model.state_dict())
        #prev_idxs = idxs
        print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
        new_idxs, grads_idxs = setf_model.naive_greedy_max(bud,list(remainList), clone_dict)
        new_idxs = list(np.array(list(remainList))[new_idxs])
        remainList = remainList.difference(new_idxs)
        idxs.extend(new_idxs)
        print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
        model.load_state_dict(cached_state_dict)
        ### Change the subset_trnloader according to new found indices: subset_idxs
        subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                       sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                       pin_memory=True)
        #print(len(list(set(prev_idxs).difference(set(idxs)))) + len(list(set(idxs).difference(set(prev_idxs)))))
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
   
    return val_accies, test_accies, unlab_accies, idxs




start_idxs = np.random.choice(N, size=no_points, replace=False)
random_subset_idx = [trainset.indices[x] for x in start_idxs]
facility_loaction_warm_start = start_idxs
#run_stochastic_Facloc(x_trn, y_trn, no_points)
#facloc_idxs = [trainset.indices[x] for x in facloc_idxs]

# Facility Location Run
#fva, fta, fua, facloc_idxs = active_learning_taylor('Facility Location',facility_loaction_warm_start,no_points)
#knn
kva, kta,kua, knn_subset_idx = active_learning_taylor('FASS',facility_loaction_warm_start)
#Badge
bva, bta,bua, badge_subset_idx = return_accuracies(facility_loaction_warm_start ,no_select-1,no_points,num_epochs,\
    learning_rate,datadir,data_name,'dss')
# Random Run
rva, rta,rua, random_subset_idx= active_learning_taylor('Random',facility_loaction_warm_start )
# Online algo run
t_va, t_ta, t_ua, subset_idxs = act_learn_online('Taylor',facility_loaction_warm_start, no_points)
#Facility Location OneStep Runs
#facloc_reg_t_va, facloc_reg_t_ta, facloc_reg_t_ua, facloc_reg_subset_idxs = train_model_Facloc(facility_loaction_warm_start, no_points,)
#Randomized Greedy Taylor OneStep Runs
rand_t_va, rand_t_ta, rand_t_ua, rand_subset_idxs = act_learn_online('Random Greedy',facility_loaction_warm_start, no_points,0.9)

plot_start_epoch = 0
x_axis = (np.arange(plot_start_epoch,no_select)+1)*no_points

###### Subset Trn loss with val = VAL #############
plt.figure()
plt.plot(x_axis, t_va[plot_start_epoch:], 'b-', label='tay_v=val',marker='o')
plt.plot(x_axis, kva[plot_start_epoch:], '#8c564b', label='FASS',marker='o')
plt.plot(x_axis, rva[plot_start_epoch:], 'g-', label='random',marker='o')
#plt.plot(x_axis, fva[plot_start_epoch:], 'pink', label='FacLoc',marker='o')
plt.plot(x_axis, rand_t_va[plot_start_epoch:], 'k-', label='rand_reg_tay_v=val',marker='o')
#plt.plot(x_axis, facloc_reg_t_va[plot_start_epoch:], 'y', label='facloc_reg_tay_v=val',marker='o')
#plt.plot(x_axis, ft_va[plot_start_epoch:], 'orange', label='NON-tay_v=val',marker='o')
plt.plot(x_axis, bva[plot_start_epoch:], 'c', label='BADGE',marker='o')

#plt.plot(x_axis, lo_va[plot_start_epoch:], 'm-', label='logit_tay_v=val')
#plt.plot(x_axis, dtay_fval_substrn_losses[plot_start_epoch:], '#8c564b', label='deep_tay_v=val')
#plt.plot(x_axis, pt_va[plot_start_epoch:], '#DD4477', label='proximal_v=val')
# plt.plot(np.arange(1,num_epochs+1), knn_fval_substrn_losses, 'r-', label='knn_v=val')

plt.legend()
plt.xlabel('No of points')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'val_acc_v=VAL.png'
plt.savefig(plt_file)
plt.clf()


########################################################################
###### Full Trn loss with val = VAL #############
plt.figure()
plt.plot(x_axis, t_ta[plot_start_epoch:], 'b-', label='tay_v=val',marker='o')
plt.plot(x_axis, kta[plot_start_epoch:], '#8c564b', label='FASS',marker='o')
plt.plot(x_axis, rta[plot_start_epoch:], 'g-', label='random',marker='o')
#plt.plot(x_axis, fta[plot_start_epoch:], 'pink', label='FacLoc',marker='o')
plt.plot(x_axis, rand_t_ta[plot_start_epoch:], 'k-', label='rand_reg_tay_v=val',marker='o')
#plt.plot(x_axis, facloc_reg_t_ta[plot_start_epoch:], 'y', label='facloc_reg_tay_v=val',marker='o')
#plt.plot(x_axis, ft_ta[plot_start_epoch:], 'orange', label='NON-tay_v=val',marker='o')
plt.plot(x_axis, bta[plot_start_epoch:], 'c', label='BADGE',marker='o')

#plt.plot(x_axis, lo_ta[plot_start_epoch:], 'm-', label='logit_tay_v=val')
#plt.plot(x_axis, dtay_fval_substrn_losses[plot_start_epoch:], '#8c564b', label='deep_tay_v=val')
#plt.plot(x_axis, pt_ta[plot_start_epoch:], '#DD4477', label='proximal_v=val')
# plt.plot(np.arange(1,num_epochs+1), knn_fval_substrn_losses, 'r-', label='knn_v=val')


plt.legend()
plt.xlabel('No of points')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'test_acc_v=VAL.png'
plt.savefig(plt_file)
plt.clf()


########################################################################
###### Validation loss with val = VAL #############
plt.figure()
plt.plot(x_axis, t_ua[plot_start_epoch:], 'b-', label='tay_v=val',marker='o')
plt.plot(x_axis, kva[plot_start_epoch:], '#8c564b', label='FASS',marker='o')
plt.plot(x_axis, rua[plot_start_epoch:], 'g-', label='random',marker='o')
#plt.plot(x_axis, fua[plot_start_epoch:], 'pink', label='FacLoc',marker='o')
plt.plot(x_axis, rand_t_ua[plot_start_epoch:], 'k-', label='rand_reg_tay_v=val',marker='o')
#plt.plot(x_axis, facloc_reg_t_ua[plot_start_epoch:], 'y', label='facloc_reg_tay_v=val',marker='o')
#plt.plot(x_axis, ft_ua[plot_start_epoch:], 'orange', label='NON-tay_v=val',marker='o')
plt.plot(x_axis, bua[plot_start_epoch:], 'c', label='BADGE',marker='o')

#plt.plot(x_axis, lo_ua[plot_start_epoch:], 'm-', label='logit_tay_v=val')
#plt.plot(x_axis, dtay_fval_substrn_losses[plot_start_epoch:], '#8c564b', label='deep_tay_v=val')
#plt.plot(x_axis, pt_ua[plot_start_epoch:], '#DD4477', label='proximal_v=val')
# plt.plot(np.arange(1,num_epochs+1), knn_fval_substrn_losses, 'r-', label='knn_v=val')


plt.legend()
plt.xlabel('No of points')
plt.ylabel('Unlabeled data Accuracy')
plt.title('Unlabeled data Accuracy ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'unlabeled_acc_v=VAL.png'
plt.savefig(plt_file)
plt.clf()

logfile = open(path_logfile, 'w')
print("=======================================", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)
print(data_name,":Budget = ", fraction, file=logfile)

methods=["One Step Taylor",'Rand Reg One Step','FASS',\
'Random',"BADGE"] #,"Full One Step" #'One step Perturbation' #,'Fac Loc Reg One Step' #'Facility Location'
val_acc =[t_va,rand_t_va,kva,rva,bva] #,ft_va #facloc_reg_t_va, #fva,
tst_acc =[t_ta,rand_t_ta,kta,rta,bta] #,ft_ta #facloc_reg_t_ta, fva,
unlabel_acc =[t_ua,rand_t_ua,kua,rua,bua] #,ft_ua #facloc_reg_t_ua fua,

print("Validation Accuracy",file=logfile)
print('---------------------------------------------------------------------',file=logfile)

title = '|Algo                            |'
for i in x_axis:
    title  = title +str(i) +"|"
print(title,file=logfile)

for m in range(len(methods)):
    title = '| '+methods[m]+' |'
    for i in range(len(x_axis)):
        title  = title +str(val_acc[m][i]) +"|"
    print(title,file=logfile)

print('---------------------------------------------------------------------',file=logfile)

print("\n", file=logfile)
print("Test Accuracy",file=logfile)
print('---------------------------------------------------------------------',file=logfile)

title = '|Algo                            |'
for i in x_axis:
    title  = title +str(i) +"|"
print(title,file=logfile)

for m in range(len(methods)):
    title = '| '+methods[m]+' |'
    for i in range(len(x_axis)):
        title  = title +str(tst_acc[m][i]) +"|"
    print(title,file=logfile)

print('---------------------------------------------------------------------',file=logfile)

print("\n", file=logfile)
print("Unlabeled data Accuracy",file=logfile)
print('---------------------------------------------------------------------',file=logfile)

title = '|Algo                            |'
for i in x_axis:
    title  = title +str(i) +"|"
print(title,file=logfile)

for m in range(len(methods)):
    title = '| '+methods[m]+' |'
    for i in range(len(x_axis)):
        title  = title +str(unlabel_acc[m][i]) +"|"
    print(title,file=logfile)

print('---------------------------------------------------------------------',file=logfile)

logfile.close()


'''mod_subset_idxs = list(mod_subset_idxs)
#print(len(mod_subset_idxs))
with open(all_logs_dir+'/mod_one_step_subset_selected.txt', 'w') as log_file:
    print(mod_subset_idxs, file=log_file)'''

subset_idxs = list(subset_idxs)
with open(all_logs_dir+'/one_step_subset_selected.txt', 'w') as log_file:
    print(subset_idxs, file=log_file)

'''psubset_idxs = list(psubset_idxs)
with open(all_logs_dir+'/one_step_proximal_subset_selected.txt', 'w') as log_file:
    print(psubset_idxs, file=log_file)

fsubset_idxs = list(fsubset_idxs)
with open(all_logs_dir+'/without_taylor_subset_selected.txt', 'w') as log_file:
    print(fsubset_idxs, file=log_file)'''

'''rsubset_idxs = list(rsubset_idxs)
with open(all_logs_dir+'/taylor_logit_subset_selected.txt', 'w') as log_file:
    print(rsubset_idxs, file=log_file)

dsubset_idxs = list(dsubset_idxs)
with open(all_logs_dir+'/taylor_logit_subset_selected.txt', 'w') as log_file:
    print(dsubset_idxs, file=log_file)'''

knn_subset_idx = list(knn_subset_idx)
with open(all_logs_dir+'/fass_subset_selected.txt', 'w') as log_file:
    print(knn_subset_idx, file=log_file)

badge_subset_idx = list(badge_subset_idx)
with open(all_logs_dir+'/badge_subset_selected.txt', 'w') as log_file:
    print(badge_subset_idx, file=log_file)

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
