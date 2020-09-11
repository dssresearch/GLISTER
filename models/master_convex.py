import copy
import datetime
import math
import numpy as np
import os
import pandas as pd
import subprocess
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from custom_dataset import *
from custom_dataset import write_knndata
# from models.mnist_net import MnistNet
from logistic_regression import LogisticRegNet
from resnet import *
from set_function_all import SetFunctionTaylor, SetFunctionBatch as SetFunction, SetFunctionCRAIG
from sklearn.model_selection import train_test_split


def plot_current_subset(x_trn,y_trn):

    com = np.append(np.reshape(y_trn, (-1, 1)), x_trn, axis=1)
    df = pd.DataFrame(com)

    class_wise = df.groupby(0).median().transpose()

    #class_wise.to_csv(str(fraction)+'_'+data_name+'.csv')

    fig = class_wise.plot.line(title=str(fraction)+'_'+data_name).get_figure()
    fig.savefig(str(fraction)+'_'+data_name+'_median.jpg')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using Device:", device)

## Convert to this argparse 
'''datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
select_every = int(sys.argv[5])
warm_method = int(sys.argv[6])  # whether to use warmstart-onestep (1) or online (0)
num_runs = int(sys.argv[7])    # number of random runs'''

datadir = './data/sensorless'#sensit_seismic'
data_name = 'sensorless'
#fraction = 0.2
num_epochs = 200
select_every = 50
warm_method = 0  
num_runs = 1

learning_rate = 0.1

if warm_method == 0:
    path_logfile = './Results/Temp/' + data_name + '.txt'
elif warm_method == 1:
    path_logfile = './Results/Temp/' + data_name + '_warm.txt'
logfile = open(path_logfile, 'w')


fullset, testset, num_cls = load_dataset_numpy(datadir, data_name)
if data_name == 'mnist':    
    x_trn, y_trn = fullset.data.float(), fullset.targets
    x_tst, y_tst = testset.data.float(), testset.targets
    x_trn = x_trn.view(x_trn.shape[0], -1)
    x_tst = x_tst.view(x_tst.shape[0], -1)
    # Get validation data: Its 10% of the entire (full) training data
    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
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

print(data_name)
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
#print(y_trn.shape, y_val.shape, y_tst.shape)

N, M = x_trn.shape

# Transfer all the data to GPU
d_t = time.time()
x_trn, y_trn = x_trn.to(device), y_trn.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)
print("Transferred data to device in time:", time.time()-d_t)
print_every = 50





## This function requires the executable to be: `../build/KNNSubmod`
## Also make a directory `./run_data/` before starting the runs.
def perform_knnsb_selection(datdir, dset_name, budget, selUsing):
    write_knndata(datadir, dset_name)
    trn_filepath = os.path.join(datadir, 'knn_' + dset_name + '.trn')
    # val_filepath = os.path.join(datadir, 'knn_' + dset_name + '.val')
    # tst_filepath = os.path.join(datadir, 'knn_' + dset_name + '.tst')
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



def train_model_knnsb(start_rand_idxs):
    ## NOTE that start idxs are NOT random. 
    ## They are given from the knnsb function.
    ## Using the var name `start_rand_idxs` just to keep copied code intact.
    model = LogisticRegNet(M, num_cls)
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
    print("Starting KnnSB Training Run!")
    for i in range(1, num_epochs+1):    
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets) 
        loss.backward()
        optimizer.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            with torch.no_grad():
                # val_in, val_t = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                full_trn_outputs = model(x_trn)
                full_trn_loss = criterion(full_trn_outputs, y_trn)
            print('Epoch:', i, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

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

    print("KNNsb Run---------------------------------",)
    print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item())
    print("Validation Loss and Accuracy:", val_loss.item(), val_acc )
    print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss.item(), test_loss.item()


def train_model_random(start_rand_idxs):
   
    model = LogisticRegNet(M, num_cls)
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
    print("Starting Random Run!")
    for i in range(1, num_epochs+1):    
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets) 
        loss.backward()
        optimizer.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            with torch.no_grad():
                # val_in, val_t = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                full_trn_outputs = model(x_trn)
                full_trn_loss = criterion(full_trn_outputs, y_trn)
            print('Epoch:', i, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

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
    return val_acc, tst_acc, val_loss.item(), test_loss.item()


def train_model_online_onestep(start_rand_idxs):

    model = LogisticRegNet(M, num_cls)
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

    setf_model = SetFunction(x_trn, y_trn, x_val, y_val, model, 
        criterion, criterion_nored, learning_rate)

    print("Starting Online OneStep Run!")
    for i in range(1, num_epochs+1):    
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)    
        loss.backward()
        optimizer.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            with torch.no_grad():
                # val_in, val_t = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                full_trn_outputs = model(x_trn)
                full_trn_loss = criterion(full_trn_outputs, y_trn)       
            print('Epoch:', i, 'SubsetTrn,allTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if (fraction != 1) and (i % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            #print("Without Taylor approximation",file=logfile)
            #print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs, grads_idxs = setf_model.naive_greedy_max(bud, clone_dict)
            idxs = new_idxs     # update the current set
            #print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
            print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
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
    return val_acc, tst_acc, val_loss.item(), test_loss.item()


def train_model_online_taylor(start_rand_idxs,bud,valid):

    model = LogisticRegNet(M, num_cls)
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

    setf_model = SetFunctionTaylor(x_trn, y_trn, x_val, y_val, valid, model, 
        criterion, criterion_nored, learning_rate,device)

    print("Starting Online OneStep Run with taylor!")
    if valid:
        print("Online OneStep Run with Taylor approximation and with Validation Set",file=logfile)
    else:
        print("Online OneStep Run with Taylor approximation and without Validation Set",file=logfile)
    
    for i in range(1, num_epochs+1):    
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        inputs, targets = x_trn[idxs], y_trn[idxs]
        optimizer.zero_grad()
        scores = model(inputs)
        loss = criterion(scores, targets)    
        loss.backward()
        optimizer.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            with torch.no_grad():
                # val_in, val_t = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                full_trn_outputs = model(x_trn)
                full_trn_loss = criterion(full_trn_outputs, y_trn)       
            print('Epoch:', i, 'SubsetTrn,allTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        if (fraction != 1) and (i % select_every == 0):
            # val_in, val_t = x_val.to(device), y_val.to(device)  # Transfer them to device
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            # Dont put the logs for Selection on logfile!!
            #print("With Taylor approximation",file=logfile)
            #print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
            t_ng_start = time.time()
            new_idxs, grads_idxs = setf_model.naive_greedy_max(bud, clone_dict)
            idxs = new_idxs     # update the current set
            #print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
            print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
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
    return val_acc, tst_acc, val_loss.item(), test_loss.item()


def CRAIG(bud):

    #trainset, testset, dimen,num_cls =  load_dataset_pytorch(datadir, data_name)

    train_batch_size = 1200
    #test_batch_size = 800
    #num_workers = 2

    #train_full_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size ,shuffle=False, num_workers=num_workers)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size ,num_workers=num_workers)

    train_full_loader = []
    for item in range(math.ceil(len(x_trn)/train_batch_size)):
      inputs = x_trn[item*train_batch_size:(item+1)*train_batch_size]
      target  = y_trn[item*train_batch_size:(item+1)*train_batch_size]
      train_full_loader.append((inputs,target))

    model = LogisticRegNet(M, num_cls)
    # if data_name == 'mnist':
    #     model = MnistNet()
    if torch.cuda.device_count() > 1:
       print("Using:", torch.cuda.device_count(), "GPUs!")
       model = nn.DataParallel(model)
       cudnn.benchmark = True
    model = model.to(device)
    

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    setf_model = SetFunctionCRAIG(device ,train_full_loader, True)

    print("Starting CRAIG!")
    print("CRAIG",file=logfile)

    if fraction != 1:
        t_ng_start = time.time()
        idxs, gamma = setf_model.lazy_greedy_max(bud,model)
        #print(gamma,file=logfile)
        plot_current_subset(x_trn[idxs].cpu().data.numpy(),y_trn[idxs].cpu().data.numpy())
        print("Lazy greedy total time with CRAIG:", time.time()-t_ng_start,file=logfile)
    else:
        idxs = [i for i in range(len(x_trn))]
        plot_current_subset(x_trn.cpu().data.numpy(),y_trn.cpu().data.numpy())
    
    
    for i in range(1, num_epochs+1):    
        # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
        if fraction != 1:
            inputs, targets = x_trn[idxs], y_trn[idxs]
        else:
            inputs, targets = x_trn, y_trn
        optimizer.zero_grad()
        scores = model(inputs)  

        if fraction != 1:
            loss = criterion_nored(scores, targets)  
            mult = torch.tensor(gamma, dtype=torch.float32).to(device) #[i*train_batch_size:(i+1)*train_batch_size]
            loss = (mult*loss).mean()
        else:
            loss = criterion(scores, targets)  
        
        loss.backward()
        optimizer.step()

        if i % print_every == 0:  # Print Training and Validation Loss
            with torch.no_grad():
                # val_in, val_t = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                full_trn_outputs = model(x_trn)
                full_trn_loss = criterion(full_trn_outputs, y_trn)       
            print('Epoch:', i, 'SubsetTrn,allTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item())

        '''if (fraction != 1) and (i % select_every == 0 or i == 5):
            
            t_ng_start = time.time()
            new_idxs, gamma = setf_model.lazy_greedy_max(bud,model)
            idxs = new_idxs     # update the current set
            print("Lazy greedy total time with CRAIG:", time.time()-t_ng_start,file=logfile)'''


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
    return val_acc, tst_acc, val_loss.item(), test_loss.item()


for fraction in [0.1, 0.2, 0.4, 0.75, 0.9,1]:

    bud = int(fraction * N)
    print("Budget, fraction and N:", bud, fraction, N)
    
    r_valacc = np.zeros(num_runs)
    r_tstacc = np.zeros(num_runs)
    r_valloss = np.zeros(num_runs)
    r_tstloss = np.zeros(num_runs)

    '''valacc = np.zeros(num_runs)
    tstacc = np.zeros(num_runs)
    valloss = np.zeros(num_runs)
    tstloss = np.zeros(num_runs)'''

    c_valacc = np.zeros(num_runs)
    c_tstacc = np.zeros(num_runs)
    c_valloss = np.zeros(num_runs)
    c_tstloss = np.zeros(num_runs)

    t_valacc = np.zeros(num_runs)
    t_tstacc = np.zeros(num_runs)
    t_valloss = np.zeros(num_runs)
    t_tstloss = np.zeros(num_runs)

    t_val_valacc = np.zeros(num_runs)
    t_val_tstacc = np.zeros(num_runs)
    t_val_valloss = np.zeros(num_runs)
    t_val_tstloss = np.zeros(num_runs)

    ## KnnSB selection with Flag = TRN and FLAG = VAL
    knn_idxs_flag_trn = perform_knnsb_selection(datadir, data_name, fraction, selUsing='trn')
    knn_idxs_flag_val = perform_knnsb_selection(datadir, data_name, fraction, selUsing='val')

    ## Training with KnnSB idxs with Flag = TRN and FLAG = VAL
    knn_valacc_flagtrn, knn_tstacc_flagtrn, knn_valloss_flagtrn, knn_tstloss_flagtrn = train_model_knnsb(knn_idxs_flag_trn)
    knn_valacc_flagval, knn_tstacc_flagval, knn_valloss_flagval, knn_tstloss_flagval = train_model_knnsb(knn_idxs_flag_val)

    for i in range(num_runs):
        
        run_start_idxs = np.random.randint(N, size=bud)

        exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) +  \
        '_selEvery:' + str(select_every) + '_variant' + str(warm_method) + '_runs' + str(num_runs)

        exp_start_time = datetime.datetime.now()
        print("=======================================", file=logfile)
        print(exp_name, str(exp_start_time), file=logfile)

        print('-----------------------------------------')
        print(exp_name, str(exp_start_time))

        r_valacc[i], r_tstacc[i], r_valloss[i], r_tstloss[i] = train_model_random(run_start_idxs)

        #valacc[i], tstacc[i], valloss[i], tstloss[i] = train_model_online_onestep(run_start_idxs,bud)
    t_valacc[i], t_tstacc[i], t_valloss[i], t_tstloss[i] = train_model_online_taylor(run_start_idxs,bud,False)
    t_val_valacc[i], t_val_tstacc[i], t_val_valloss[i], t_val_tstloss[i] = train_model_online_taylor(run_start_idxs,bud,True)
    
    c_valacc[i], c_tstacc[i], c_valloss[i], c_tstloss[i] = CRAIG(bud) 
        
    '''v1, std_v1 = np.mean(valacc), np.std(valacc)
    v2, std_v2 = np.mean(valloss), np.std(valloss)
    t1, std_t1 = np.mean(tstacc), np.std(tstacc)
    t2, std_t2 = np.mean(tstloss), np.std(tstloss)'''

    cv1, cstd_v1 = np.mean(c_valacc), np.std(c_valacc)
    cv2, cstd_v2 = np.mean(c_valloss), np.std(c_valloss)
    ct1, cstd_t1 = np.mean(c_tstacc), np.std(c_tstacc)
    ct2, cstd_t2 = np.mean(c_tstloss), np.std(c_tstloss)

    rv1, rstd_v1 = np.mean(r_valacc), np.std(r_valacc)
    rv2, rstd_v2 = np.mean(r_valloss), np.std(r_valloss)
    rt1, rstd_t1 = np.mean(r_tstacc), np.std(r_tstacc)
    rt2, rstd_t2 = np.mean(r_tstloss), np.std(r_tstloss)

    tv1, tstd_v1 = np.mean(t_valacc), np.std(t_valacc)
    tv2, tstd_v2 = np.mean(t_valloss), np.std(t_valloss)
    tt1, tstd_t1 = np.mean(t_tstacc), np.std(t_tstacc)
    tt2, tstd_t2 = np.mean(t_tstloss), np.std(t_tstloss)

    tvv1, tstd_vv1 = np.mean(t_val_valacc), np.std(t_val_valacc)
    tvv2, tstd_vv2 = np.mean(t_val_valloss), np.std(t_val_valloss)
    ttv1, tstd_tv1 = np.mean(t_val_tstacc), np.std(t_val_tstacc)
    ttv2, tstd_tv2 = np.mean(t_val_tstloss), np.std(t_val_tstloss)

    #with open(path_logfile, 'a') as logfile:
    '''print("=========Online Selection===================", file=logfile)       
    print("Avg Validation ACCURACY and Std:", v1, std_v1, file=logfile)
    print("Avg Validation LOSS and Std:", v2, std_v2, file=logfile)
    print("Avg Test Data ACCURACY and Std:", t1, std_t1, file=logfile)
    print("Avg Test Data LOSS and Std:", t2, std_t2, file=logfile)'''
    print(data_name,":Budget = ",fraction,file=logfile)
    print('---------------------------------------------------------------------',file=logfile)
    print('|Algo                            | Val Acc       |   Test Acc       |',file=logfile)
    print('| -------------------------------|:-------------:| ----------------:|',file=logfile)
    print('*| Craig                          |',cv1,',',cstd_v1,'  | ',ct1, cstd_t1,' |',file=logfile)
    print('*| Taylor without Validation Set  |',tv1,',',tstd_v1,'  | ',tt1, tstd_t1,' |',file=logfile)
    print('*| Taylor with Validation Set     |',tvv1,',',tstd_vv1,'| ',ttv1, tstd_tv1,' |',file=logfile)
    print('*| KnnSB with Validation=TRN      |',knn_valacc_flagtrn,'| ',knn_tstacc_flagtrn,' |',file=logfile)
    print('*| KnnSB with Validation=VAL      |',knn_valacc_flagval,'| ',knn_tstacc_flagval,' |',file=logfile)
    print('*| Random                         |',rv1,',',rstd_v1,'  | ',rt1, rstd_t1,' |',file=logfile)
    
    print("\n=========CRAIG===================", file=logfile)       
    #print("Avg Validation ACCURACY and Std:", cv1, cstd_v1, file=logfile)
    print("*Avg Validation LOSS and Std:", cv2, cstd_v2, file=logfile)
    #print("Avg Test Data ACCURACY and Std:", ct1, cstd_t1, file=logfile)
    print("*Avg Test Data LOSS and Std:", ct2, cstd_t2, file=logfile)
    print("=========Online Selection Taylor without Validation Set===================", file=logfile)      
    #print("Taylor Avg Validation ACCURACY and Std:", tv1, tstd_v1, file=logfile)
    print("*Taylor Avg Validation LOSS and Std:", tv2, tstd_v2, file=logfile)
    #print("Taylor Avg Test Data ACCURACY and Std:", tt1, tstd_t1, file=logfile)
    print("*Taylor Test Data LOSS and Std:", tt2, tstd_t2, file=logfile)
    print("=========Online Selection Taylor with Validation Set===================", file=logfile)      
    #print("Taylor Avg Validation ACCURACY and Std:", tvv1, tstd_vv1, file=logfile)
    print("*Taylor Avg Validation LOSS and Std:", tvv2, tstd_vv2, file=logfile)
    #print("Taylor Avg Test Data ACCURACY and Std:", ttv1, tstd_tv1, file=logfile)
    print("*Taylor Test Data LOSS and Std:", ttv2, tstd_tv2, file=logfile)

    print("=========kNNsb Selection with Validation = TRN ===================", file=logfile)      
    print("*KnnSB Validation LOSS:", knn_valloss_flagtrn, file=logfile)
    print("*KnnSB Test Data LOSS:", knn_tstloss_flagtrn, file=logfile)

    print("=========kNNsb Selection with Validation = VAL ===================", file=logfile)      
    print("*KnnSB Validation LOSS:", knn_valloss_flagval, file=logfile)
    print("*KnnSB Test Data LOSS:", knn_tstloss_flagval, file=logfile)

    print("--------Random Results-----------------------", file=logfile)       
    #print("Rand Avg Validation ACCURACY and Std:", rv1, rstd_v1, file=logfile)
    print("*Rand Avg Validation LOSS and Std:", rv2, rstd_v2, file=logfile)
    #print("Rand Avg Test Data ACCURACY and Std:", rt1, rstd_t1, file=logfile)
    print("*Rand Avg Test Data LOSS and Std:", rt2, rstd_t2, file=logfile)
    print("==================================================", file=logfile)       