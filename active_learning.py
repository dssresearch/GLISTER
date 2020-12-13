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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from models.simpleNN_net import * #ThreeLayerNet
from models.logistic_regression import LogisticRegNet
from models.set_function_act_learn import SetFunctionFacLoc, SetFunctionTaylor, SetFunctionBatch,\
    SDReg_GlisterActLinear_SetFunction_Closed_Vect,Small_GlisterAct_Linear_SetFunction_Closed_Vect,\
    FacReg_GlisterActLinear_SetFunction_Closed_Vect #Small_GlisterAct_Linear_SetFunction_Closed
from sklearn.model_selection import train_test_split
#from utils.custom_dataset import CustomDataset_act, load_dataset_numpy, write_knndata
#from custom_dataset_old import load_dataset_numpy as load_dataset_numpy_old, write_knndata as write_knndata_old
from utils.custom_dataset import load_dataset_custom, CustomDataset_WithId, load_mnist_cifar, write_knndata
import math
import random
from torch.distributions import Categorical

import sys
sys.path.append('./badge/')
from badge.my_run import return_accuracies

device = "cuda" if torch.cuda.is_available() else "cpu"
# #device = "cpu"
print("Using Device:", device)


datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
no_select = int(sys.argv[5])
feature = sys.argv[6]
print_every = 50

learning_rate = 0.05
if feature != 'classimb':
    all_logs_dir = './results/ActLearn/' + data_name + '/' + str(fraction) + '/' + str(no_select)
else:
    all_logs_dir = './results/ActLearn/' + data_name + '_imbalance/' + str(fraction) + '/' + str(no_select)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt') 

exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + '_no of selection:' + str(no_select) 

print(exp_name)
exp_start_time = datetime.datetime.now()


if data_name == 'mnist':

    fullset, valset, testset, num_cls = load_mnist_cifar(datadir, data_name, feature=feature)

    x_trn, y_trn = fullset.data, fullset.targets
    x_tst, y_tst = testset.data, testset.targets
    x_trn = x_trn.view(x_trn.shape[0], -1)
    x_tst = x_tst.view(x_tst.shape[0], -1)
    # Get validation data: Its 10% of the entire (full) training data
    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
else:

    fullset, valset, testset, data_dims, num_cls = load_dataset_custom(datadir, data_name,feature=feature, isnumpy=True)
    
    x_trn, y_trn = torch.from_numpy(fullset[0]).float(), torch.from_numpy(fullset[1]).long()
    x_tst, y_tst = torch.from_numpy(testset[0]).float(), torch.from_numpy(testset[1]).long()
    x_val, y_val = torch.from_numpy(valset[0]).float(), torch.from_numpy(valset[1]).long()

write_knndata(datadir, data_name,feature=feature)


print('-----------------------------------------')
print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
#print(y_trn.shape, y_val.shape, y_tst.shape)
#print("Datadir",datadir,"Data name",data_name)

N, M = x_trn.shape
#bud = int(fraction * N)
no_points = math.ceil(fraction*N/no_select)
print("No. of starting points and N:", no_points, N)
# Transfer all the data to GPU
x_trn, y_trn = x_trn.to(device), y_trn.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)

train_batch_size_for_greedy = 800#1200 
train_batch_size = 128

def predict(model, X, Y):

    '''if type(X) is np.ndarray:
        loader_te = DataLoader(CustomDataset_WithId(X, Y,transform=None),shuffle=False,\
            batch_size=1000)
    else: 
        loader_te =  DataLoader(CustomDataset_WithId(X.numpy(), Y, transform=None),\
            shuffle=False, batch_size=1000)'''

    loader_te = DataLoader(CustomDataset_WithId(X, Y,transform=None),shuffle=False,\
            batch_size=1000)

    model.eval()
    P = torch.zeros(len(Y)).long()
    with torch.no_grad():
        for batch_idx in list(loader_te.batch_sampler):
            x, y, idxs = loader_te.dataset[batch_idx]
            if 'numpy' in str(type(x)):
                x = torch.from_numpy(x).type(torch.float)
            if 'numpy' in str(type(y)):
                y = torch.from_numpy(y).type(torch.long)
            x, y = Variable(x.cuda()), Variable(y.cuda())
            out, e1 = model(x)
            pred = out.max(1)[1]
            P[idxs] = pred.data.cpu()
    return P

def perform_knnsb_selection(datadir, dset_name,X,Y, budget, selUsing):

    trndata = np.c_[X.cpu(), Y.cpu()]
    # Write out the trndata
    trn_filepath = os.path.join(datadir, 'act_'+feature+'_knn_' + dset_name + '.trn')
    np.savetxt(trn_filepath, trndata, fmt='%.6f')
    
    run_path = './knn_results/'
    output_dir = run_path + 'KNNSubmod_' + dset_name + '/'
    indices_file = output_dir + 'act_'+feature+'_KNNSubmod_' + str((int)(budget*100)) + ".subset"

    if selUsing == 'val':
        val_filepath = os.path.join(datadir, feature+'_knn_' + dset_name + '.val')
    else:
        val_filepath = trn_filepath

    subprocess.call(["mkdir","-p", output_dir])
    knnsb_args = []
    knnsb_args.append('./build/KNNSubmod')
    knnsb_args.append(trn_filepath)
    knnsb_args.append(val_filepath)
    knnsb_args.append(" ")  # File delimiter!!
    knnsb_args.append(str(no_points/X.shape[0]))
    knnsb_args.append(indices_file)
    knnsb_args.append("1")  # indicates cts data. Deprecated.
    print("Obtaining the subset")
    #print(knnsb_args)
    subprocess.run(knnsb_args)
    print("finished selection")
    # Can make it return the indices_file if using with other function. 
    idxs_knnsb = np.genfromtxt(indices_file, delimiter=',', dtype=int) # since they are indices!
    return idxs_knnsb

def gen_rand_prior_indices(remainset, size):
    per_sample_count = [len(torch.where(y_trn[remainset] == x)[0]) for x in np.arange(num_cls)]
    per_sample_budget = int(size/num_cls)
    total_set = list(np.arange(len(remainset)))
    indices = []
    count = 0
    for i in range(num_cls):
        label_idxs = torch.where(y_trn[remainset] == i)[0].cpu().numpy()
        if per_sample_count[i] > per_sample_budget:
            indices.extend(list(np.random.choice(label_idxs, size=per_sample_budget, replace=False)))
        else:
            indices.extend(label_idxs)
            count += (per_sample_budget - per_sample_count[i])
    for i in indices:
        total_set.remove(i)
    indices.extend(list(np.random.choice(total_set, size= count, replace=False)))
    return np.array(remainset)[indices]

facility_loaction_warm_start = []

def run_stochastic_Facloc(data, targets, budget):
    model = TwoLayerNet(M, num_cls, 100)
    model = model.to(device)
    approximate_error = 0.01
    per_iter_bud = 10
    num_iterations = int(budget/10)
    facloc_indices = []
    trn_indices = list(np.arange(len(data)))
    sample_size = int(len(data) / num_iterations * math.log(1 / approximate_error))
    #greedy_batch_size = 1200
    for i in range(num_iterations):
        rem_indices = list(set(trn_indices).difference(set(facloc_indices)))
        state = np.random.get_state()
        np.random.seed(i*i)
        sub_indices = np.random.choice(rem_indices, size=sample_size, replace=False)
        np.random.set_state(state)
        data_subset = data[sub_indices].cpu()
        #targets_subset = targets[sub_indices].cpu()
        #train_loader_greedy = []
        #train_loader_greedy.append((data_subset, targets_subset))
        setf_model = SetFunctionFacLoc(device,1200)
        idxs = setf_model.lazy_greedy_max(per_iter_bud,data_subset, model)
        facloc_indices.extend([sub_indices[idx] for idx in idxs])
    return facloc_indices

def active_learning_taylor(func_name,start_rand_idxs=None, bud=None, valid=True,fac_loc_idx=None):
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    #model = ThreeLayerNet(M, num_cls, 5, 5)
    #model = LogisticRegNet(M, num_cls)
    model = TwoLayerNet(M, num_cls, 100)
    # if data_name == 'mnist':
    #     model = MnistNet()
    '''if torch.cuda.device_count() > 1:
        print("Using:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True'''

    model = model.to(device)

    sub_idxs = start_rand_idxs

    '''if func_name == 'Facloc Regularized':
        x_val1 = torch.cat([x_val, x_trn[fac_loc_idx]], dim=0)
        y_val1 = torch.cat([y_val, y_trn[fac_loc_idx]], dim=0)'''

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if func_name == 'Full OneStep':
        setf_model = SetFunctionBatch(x_val, y_val, model, criterion, criterion_nored, learning_rate, device)

    elif func_name == 'Facility Location':
        if data_name != 'covertype':
            setf_model = SetFunctionFacLoc(device, train_batch_size_for_greedy)
            sub_idxs = setf_model.lazy_greedy_max(bud, x_trn,model)
        else:
            sub_idxs = run_stochastic_Facloc(x_trn, y_trn, bud)

        facility_loaction_warm_start = copy.deepcopy(sub_idxs)

    elif func_name == 'Facloc Regularized':
        setf_model = FacReg_GlisterActLinear_SetFunction_Closed_Vect(x_val, y_val, model, criterion,
                 criterion_nored, learning_rate, device, num_cls)
            #SetFunctionTaylor(x_val1, y_val1, model, criterion, criterion_nored, learning_rate, device,num_cls)

    elif func_name == 'With Diversity':
        setf_model = SDReg_GlisterActLinear_SetFunction_Closed_Vect(x_val, y_val, model, criterion,
                 criterion_nored, learning_rate, device, num_cls)
            #SetFunctionTaylor(x_val1, y_val1, model, criterion, criterion_nored, learning_rate, device,num_cls)

    else:
        #setf_model = SetFunctionTaylorDeep(train_loader_greedy, valid_loader, valid, model, 
        #        criterion, criterion_nored, learning_rate, device, N)
        setf_model = Small_GlisterAct_Linear_SetFunction_Closed_Vect(x_val, y_val, model, criterion,
                 criterion_nored, learning_rate, device, num_cls)
            #SetFunctionTaylor(x_val, y_val, model, criterion, criterion_nored, learning_rate, device,num_cls)

        #setf_model = SetFunctionTaylorDeep_ReLoss_Mean(x_trn, y_trn, train_batch_size_for_greedy, x_val, y_val, valid, model, 
        #        criterion, criterion_nored, learning_rate, device, N) 

     

    remainList = set(list(range(N)))
    sub_idxs = list(sub_idxs)
    remainList = remainList.difference(sub_idxs)

    if func_name == 'Taylor Online':
        print("Starting Online OneStep Run with taylor on loss!")
    elif func_name == 'Full OneStep':
        print("Starting Online OneStep Run without taylor!")
    elif func_name == 'Facloc Regularized':
        print("Starting Facility Location Regularized Online OneStep Run with taylor!")
    elif func_name == 'With Diversity':
        print("Starting Diversity based Online OneStep Run with taylor!")
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
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        if isinstance(m, nn.Linear):
            #m.reset_parameters()
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)

    model =  model.apply(weight_reset).cuda()
    #print(model.linear2.weight)
    for n in range(no_select):
        sub_idxs.sort()
        np.random.seed(42)
        np_sub_idxs = np.array(sub_idxs)
        np.random.shuffle(np_sub_idxs)
        loader_tr = DataLoader(CustomDataset_WithId(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
             transform=None),shuffle=False,batch_size=train_batch_size)
        model.train()
        for i in range(num_epochs):
            # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
            '''inputs, targets = x_trn[idxs], y_trn[idxs]
            optimizer.zero_grad()
            scores = model(inputs)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()'''
            #model =  model.apply(weight_reset).cuda()

            accFinal = 0. 
            for batch_idx in list(loader_tr.batch_sampler):
                x, y, idxs = loader_tr.dataset[batch_idx]

                x, y = Variable(x.cuda()), Variable(y.cuda())
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
                loss.backward()

                # clamp gradients, just in case
                for p in filter(lambda p: p.grad is not None, model.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

                optimizer.step()

            if (i % 50 == 0) and (accFinal/len(loader_tr.dataset.X) < 0.2): # reset if not converging
                model =  model.apply(weight_reset).cuda()
                optimizer = optim.SGD(model.parameters(), lr = learning_rate)

            #if accFinal/len(loader_tr.dataset.X) >= 0.99:
            #    break

            '''with torch.no_grad():
                # val_in, val_t = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                full_trn_outputs = model(x_trn)
                full_trn_loss = criterion(full_trn_outputs, y_trn)'''

            #accFinal = torch.sum((torch.max(scores,1)[1] == targets).float()).data.item()
            #print(accFinal / len(loader_tr.dataset.X))

            #if i % print_every == 0:  # Print Training and Validation Loss
        #print( n+1,'Time', 'SubsetTrn', loss.item())#, ,FullTrn,ValLoss: full_trn_loss.item(), val_loss.item())

        curr_X_trn = x_trn[list(remainList)]
        curr_Y_trn = y_trn[list(remainList)]

        """model.eval()
        with torch.no_grad():
            '''full_trn_out = model(x_trn)
            full_trn_loss = criterion(full_trn_out, y_trn).mean()
            sub_trn_out = model(x_trn[idxs])
            sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()'''

            val_out = model(x_val)
            val_loss = criterion(val_out, y_val)
            _, val_predict = val_out.max(1)
            val_correct = val_predict.eq(y_val).sum().item()
            val_total = y_val.size(0)
            val_acc = 100 * val_correct / val_total

            correct = 0
            total = 0
            
            inputs, targets = x_tst.to(device), y_tst.to(device)
            outputs = model(inputs)
            test_loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            tst_acc = 100.0 * correct / total

            rem_out = model(curr_X_trn)
            rem_loss = criterion(rem_out, curr_Y_trn)
            _, rem_predict = rem_out.max(1)
            rem_correct = rem_predict.eq(curr_Y_trn).sum().item()
            rem_total = curr_Y_trn.size(0)
            rem_acc = 100 * rem_correct / rem_total"""

        val = predict(model,x_val,y_val)
        tst = predict(model,x_tst,y_tst)
        ulab = predict(model,curr_X_trn,curr_Y_trn)
        
        val_accies[n] = 100.0 * val.eq(torch.tensor(y_val)).sum().item()/ y_val.size(0) #val_acc
        test_accies[n] = 100.0 * tst.eq(torch.tensor(y_tst)).sum().item()/ y_tst.size(0) #tst_acc
        unlab_accies[n] = 100.0 * ulab.eq(torch.tensor(curr_Y_trn)).sum().item()/ curr_Y_trn.size(0) #rem_acc

        print( n+1,'Time', 'Test acc', tst_acc)

        #if ((i + 1) % select_every == 0) and func_name not in ['Facility Location','Random']:
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
        cached_state_dict = copy.deepcopy(model.state_dict())
        clone_dict = copy.deepcopy(model.state_dict())
        # Dont put the logs for Selection on logfile!!
        # print("With Taylor approximation",file=logfile)
        # print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
        #t_ng_start = time.time()

        if func_name == 'Random Greedy':
            new_idxs = setf_model.naive_greedy_max(curr_X_trn,rem_predict,int(0.9 * no_points), clone_dict)
            new_idxs = list(np.array(list(remainList))[new_idxs])
            
            remainList = remainList.difference(new_idxs)
            new_idxs.extend(list(np.random.choice(list(remainList), size=int(0.1 * no_points), replace=False)))
            remainList = remainList.difference(new_idxs)
            sub_idxs.extend(new_idxs)

        elif func_name == "FASS":

            fn = nn.Softmax(dim=1)
            soft = fn(rem_out)

            entropy2 = Categorical(probs = soft).entropy()

            #print(entropy2.shape)
            if 5*no_points < entropy2.shape[0]:
                values,indices = entropy2.topk(5*no_points)
                #indices = list(np.array(list(remainList))[indices.cpu()])
            else:
                indices = [i for i in range(entropy2.shape[0])]#list(remainList)

            knn_idxs_flag_val = perform_knnsb_selection(datadir, data_name, curr_X_trn[indices],rem_predict[indices], 
                fraction, selUsing='val') 
            #print(knn_idxs_flag_val)
            #print(len(knn_idxs_flag_val))

            ##print(len(knn_idxs_flag_val),len(indices))
            knn_idxs_flag_val = list(np.array(list(remainList))[indices.cpu()][knn_idxs_flag_val])

            remainList = remainList.difference(knn_idxs_flag_val)
            sub_idxs.extend(knn_idxs_flag_val)

        elif func_name == 'Random':
            state = np.random.get_state()
            np.random.seed(n*n)
            #new_idxs = gen_rand_prior_indices(list(remainList), size=no_points)
            new_idxs = np.random.choice(list(remainList), size=no_points, replace=False)
            np.random.set_state(state)
            remainList = remainList.difference(new_idxs)
            sub_idxs.extend(new_idxs)


        elif func_name == 'Random Perturbation':
            new_idxs = setf_model.naive_greedy_max(curr_X_trn,rem_predict,no_points, clone_dict,None,True)  # , grads_idxs
            new_idxs = np.array(list(remainList))[new_idxs]

            remainList = remainList.difference(new_idxs)
            sub_idxs.extend(new_idxs) 

        elif func_name == 'Facility Location':

            if data_name == 'covertype':
                new_idxs = run_stochastic_Facloc(curr_X_trn, rem_predict, bud)
            else:
                new_idxs = setf_model.lazy_greedy_max(bud, curr_X_trn ,model)
            new_idxs = np.array(list(remainList))[new_idxs]

            remainList = remainList.difference(new_idxs)
            sub_idxs.extend(new_idxs)

        else: 
            new_idxs = setf_model.naive_greedy_max(curr_X_trn, rem_predict, no_points, clone_dict)  # , grads_idxs
            new_idxs = np.array(list(remainList))[new_idxs]
            remainList = remainList.difference(new_idxs)
            sub_idxs.extend(new_idxs) 

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
    
    if func_name == 'Facility Location':
        return val_accies, test_accies, unlab_accies, sub_idxs,facility_loaction_warm_start
    else:
        return val_accies, test_accies, unlab_accies, sub_idxs


start_idxs = np.random.choice(N, size=no_points, replace=False)
#facility_loaction_warm_start = start_idxs
# Facility Location Run
fva, fta, fua, facloc_idxs, facility_loaction_warm_start = active_learning_taylor('Facility Location',None,no_points)
#knn
kva, kta,kua, knn_subset_idx = active_learning_taylor('FASS',facility_loaction_warm_start)
#Badge
bva, bta,bua, badge_subset_idx = return_accuracies(facility_loaction_warm_start ,no_select-1,no_points,num_epochs,\
    learning_rate,datadir,data_name,feature)
# Random Run
rva, rta,rua, random_subset_idx= active_learning_taylor('Random',facility_loaction_warm_start )
# Online algo run
t_va, t_ta, t_ua, subset_idxs = active_learning_taylor('Taylor Online',facility_loaction_warm_start , no_points, True)
# Online with diversity algo run
dt_va, dt_ta, dt_ua, d_subset_idxs = active_learning_taylor('With Diversity',facility_loaction_warm_start , no_points, True)
#Facility Location OneStep Runs
facloc_reg_t_va, facloc_reg_t_ta, facloc_reg_t_ua, facloc_reg_subset_idxs = active_learning_taylor\
('Facloc Regularized',facility_loaction_warm_start , no_points, True,facloc_idxs)
#Randomized Greedy Taylor OneStep Runs
rand_t_va, rand_t_ta, rand_t_ua, rand_subset_idxs = active_learning_taylor('Random Greedy',facility_loaction_warm_start \
    , no_points, True)
# Full version Online algo run
#ft_va, ft_ta, ft_ua, fsubset_idxs = active_learning_taylor('Full OneStep',facility_loaction_warm_start , no_points, True)

# Online on Logit algo run
#lo_va, lo_ta, lo_ua, lo_subset_idxs = active_learning_taylor('Taylor on Logit',start_idxs, bud, True)
# Proximal Online algo run
#pt_va, pt_ta, pt_ua, psubset_idxs = active_learning_taylor('Proximal',start_idxs, bud, True)


plot_start_epoch = 0
x_axis = (np.arange(plot_start_epoch,no_select)+1)*no_points

###### Subset Trn loss with val = VAL #############
plt.figure()
plt.plot(x_axis, t_va[plot_start_epoch:], 'b-', label='tay_v=val',marker='o')
plt.plot(x_axis, kva[plot_start_epoch:], '#8c564b', label='FASS',marker='o')
plt.plot(x_axis, rva[plot_start_epoch:], 'g-', label='random',marker='o')
plt.plot(x_axis, fva[plot_start_epoch:], 'pink', label='FacLoc',marker='o')
plt.plot(x_axis, rand_t_va[plot_start_epoch:], 'k-', label='rand_reg_tay_v=val',marker='o')
plt.plot(x_axis, facloc_reg_t_va[plot_start_epoch:], 'y', label='facloc_reg_tay_v=val',marker='o')
#plt.plot(x_axis, ft_va[plot_start_epoch:], 'orange', label='NON-tay_v=val',marker='o')
plt.plot(x_axis, dt_va[plot_start_epoch:], 'orange', label='Diversity One Step',marker='o')
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
plt.plot(x_axis, fta[plot_start_epoch:], 'pink', label='FacLoc',marker='o')
plt.plot(x_axis, rand_t_ta[plot_start_epoch:], 'k-', label='rand_reg_tay_v=val',marker='o')
plt.plot(x_axis, facloc_reg_t_ta[plot_start_epoch:], 'y', label='facloc_reg_tay_v=val',marker='o')
#plt.plot(x_axis, ft_ta[plot_start_epoch:], 'orange', label='NON-tay_v=val',marker='o')
plt.plot(x_axis, dt_ta[plot_start_epoch:], 'orange', label='Diversity One Step',marker='o')
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
plt.plot(x_axis, kua[plot_start_epoch:], '#8c564b', label='FASS',marker='o')
plt.plot(x_axis, rua[plot_start_epoch:], 'g-', label='random',marker='o')
plt.plot(x_axis, fua[plot_start_epoch:], 'pink', label='FacLoc',marker='o')
plt.plot(x_axis, rand_t_ua[plot_start_epoch:], 'k-', label='rand_reg_tay_v=val',marker='o')
plt.plot(x_axis, facloc_reg_t_ua[plot_start_epoch:], 'y', label='facloc_reg_tay_v=val',marker='o')
#plt.plot(x_axis, ft_ua[plot_start_epoch:], 'orange', label='NON-tay_v=val',marker='o')
plt.plot(x_axis, dt_ua[plot_start_epoch:], 'orange', label='Diversity One Step',marker='o')
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

methods=["One Step Taylor",'Fac Loc Reg One Step','Rand Reg One Step','Facility Location','FASS',\
'Random',"BADGE","Diversity One Step"] #,"Full One Step" #'One step Perturbation'
val_acc =[t_va,facloc_reg_t_va,rand_t_va,fva,kva,rva,bva,dt_va] #,ft_va
tst_acc =[t_ta,facloc_reg_t_ta,rand_t_ta,fta,kta,rta,bta,dt_ta] #,ft_ta
unlabel_acc =[t_ua,facloc_reg_t_ua,rand_t_ua,fua,kua,rua,bua,dt_ua] #,ft_ua

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

'''subset_idxs = list(subset_idxs)
with open(all_logs_dir+'/one_step_subset_selected.txt', 'w') as log_file:
    print(subset_idxs, file=log_file)

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
    print(facloc_idxs, file=log_file)'''
