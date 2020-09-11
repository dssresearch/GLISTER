import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from models.set_function_all import SetFunctionFacLoc
from models.set_function_all import SetFunctionTaylorDeep
from models.simpleNN_net import *
# from resnet import *
# from models.mnist_net import MnistNet
from utils.custom_dataset import *
from utils.custom_dataset import write_knndata

torch.manual_seed(42)
np.random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
#device = "cpu"
print("Using Device:", device)

## Convert to this argparse 

datadir = sys.argv[1]
data_name = sys.argv[2]
#datadir = "./data/dna"
#data_name = "dna"
# fraction = float(sys.argv[3])
# num_epochs = int(sys.argv[4])
# select_every = int(sys.argv[5])

# datadir = './data/sensorless'#sensit_seismic'
# data_name = 'sensorless'
# fraction = 0.1  ## Dummy var. Not Used
num_epochs = 200
select_every = 70
print_every = 40

# whether to also compare against the older, slower but perhaps more accurate method
# if it is false, we just assign dummy zero values to metrics of old method.
compare_method = True  

learning_rate = 0.1
hidden1 = 100
hidden2 = 100

# all_logs_dir = './results/temp/' + data_name
all_logs_dir = './results/NN_sep/' + data_name
print(all_logs_dir)
subprocess.run(["mkdir", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt') 
logfile = open(path_logfile, 'w')

#fullset, testset, num_cls = load_dataset_numpy(datadir, data_name)
trainset, valset, testset, data_dims, num_cls = load_dataset_pytorch_sep_val(datadir, data_name,device)

train_batch_size = 128
train_batch_size_for_greedy = 1200
'''test_batch_size = 32
valid_size = 0.1
num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
valid_set = torch.utils.data.Subset(trainset, valid_idx)
train_set = torch.utils.data.Subset(trainset, train_idx)'''
print(data_dims)
M = data_dims
N = len(trainset)
print(N)

kwargs = {'num_workers': 2, 'pin_memory': True} if device=='cuda' else {}

#num_workers =2 
valid_loader = torch.utils.data.DataLoader(valset, batch_size=len(valset),shuffle=False,sampler=None,**kwargs)#num_workers=num_workers)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=N ,shuffle=False,sampler=None,**kwargs)#num_workers=num_workers)
train_loader_greedy = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size_for_greedy ,shuffle=False, sampler=None,**kwargs)#num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset) ,**kwargs)#num_workers=num_workers)



print(data_name)
write_knndata(datadir, data_name)
#shuffling
#rnd_seed = 42
#torch.manual_seed(rnd_seed)
#np.random.seed(rnd_seed)
#np.random.shuffle(indices)
#
#x_trn, y_trn = fullset
#x_tst, y_tst = testset
#
## Load as a Pytorch Tensor
#x_trn = torch.from_numpy(x_trn.astype(np.float32))
#x_tst = torch.from_numpy(x_tst.astype(np.float32))
#y_trn = torch.from_numpy(y_trn.astype(np.int64))
#y_tst = torch.from_numpy(y_tst.astype(np.int64))
## Get validation data: Its 10% of the entire (full) training data
#x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

# MNIST CODE NOT NEEDED HERE
# if data_name == 'mnist':    
#     x_trn, y_trn = fullset.data.float(), fullset.targets
#     x_tst, y_tst = testset.data.float(), testset.targets
#     x_trn = x_trn.view(x_trn.shape[0], -1)
#     x_tst = x_tst.view(x_tst.shape[0], -1)
#     # Get validation data: Its 10% of the entire (full) training data
#     x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
# else:


#print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
#print(y_trn.shape, y_val.shape, y_tst.shape)

#N, M = x_trn.shape

# Transfer all the data to GPU
#d_t = time.time()
#x_trn, y_trn = x_trn.to(device), y_trn.to(device)
#x_val, y_val = x_val.to(device), y_val.to(device)
#print("Transferred data to device in time:", time.time()-d_t)


# Write out the knn data only once before the selection for various fractions.


## This function requires the executable to be: `../build/KNNSubmod`
## Also make a directory `./run_data/` before starting the runs.
def perform_knnsb_selection(datdir, dset_name, budget, selUsing):
    # write_knndata(datadir, dset_name)
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

def train_model(set_function,start_rand_idxs, bud=None, valid=None):

    torch.manual_seed(42)
    np.random.seed(42)

    #####   MODEL  ############
    #model = LogisticRegNet(M, num_cls)
    model =  TwoLayerNet(data_dims, num_cls,hidden1)
    #model =  ThreeLayerNet(data_dims, num_cls,hidden1,hidden2)
    #######################

    if torch.cuda.device_count() > 1 and device == "cuda":
       print("Using:", torch.cuda.device_count(), "GPUs!")
       model = nn.DataParallel(model)
       cudnn.benchmark = True
    
    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    if set_function == SetFunctionFacLoc:
        setf_model = SetFunctionFacLoc(device ,train_loader_greedy)
        if fraction != 1:
            idxs= setf_model.lazy_greedy_max(bud,model)   
        print("Starting Facility Location!")
    elif set_function == SetFunctionTaylorDeep:
        setf_model = SetFunctionTaylorDeep(train_loader, valid_loader, valid, model, 
            criterion, criterion_nored, learning_rate, device, len(trainset))
        print("Starting Online OneStep Run with taylor!")
    elif set_function == "KNNSB":
        print("Starting KnnSB Training Run!")
    elif set_function == "Random":
        print("Starting Random Run!")
    elif set_function == "Online-Random":
        print("Starting Online-Random Run!")
    
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs+1)
    val_losses = np.zeros(num_epochs)
    #idxs = start_rand_idxs

    if fraction != 1:
        #train_sub_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, 
                #shuffle=False, sampler=SequentialSampler(idxs), num_workers=2)
        train_sub_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, idxs),batch_size=train_batch_size, 
                shuffle=False, **kwargs)#num_workers=2)
    else:
        train_sub_loader = train_loader

    model.train()
    for i in range(num_epochs):    
        total_subset_loss = 0 
    
        for idx, data in  enumerate(train_sub_loader, 0):    
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
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
                inputs, target = inputs.to(device), target.to(device)
                scores = model(inputs)
                tr_loss = criterion(scores, target)
                total_train_loss += tr_loss.item()

            full_trn_loss = (1.0*total_train_loss)/(idx+1)
            
            total_valid_loss = 0
            for idx, val_data in  enumerate(valid_loader, 0):
                inputs, target = val_data
                inputs, target = inputs.to(device), target.to(device)
                scores = model(inputs)
                val_loss = criterion(scores, target)
                total_valid_loss += val_loss.item()
            
            val_loss = (1.0*total_valid_loss)/(idx+1)
            
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss

        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i, 'SubsetTrn,FullTrn,ValLoss:', substrn_losses[i], full_trn_loss, val_loss)

        if ((i+1) % select_every == 0 and fraction != 1):

            if set_function == SetFunctionTaylorDeep:
                cached_state_dict = copy.deepcopy(model.state_dict())
                clone_dict = copy.deepcopy(model.state_dict())
                # Dont put the logs for Selection on logfile!!
                #print("With Taylor approximation",file=logfile)
                #print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()),file=logfile)
                t_ng_start = time.time()
                new_idxs = setf_model.naive_greedy_max(bud, clone_dict) #, grads_idxs
                idxs = new_idxs     # update the current set

                train_sub_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, idxs),batch_size=train_batch_size, 
                    shuffle=False, **kwargs)#num_workers=2)
                #print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()),file=logfile)
                # print("Naive greedy total time with taylor:", time.time()-t_ng_start,file=logfile)
                model.load_state_dict(cached_state_dict)
                    
            elif set_function == "Online-Random":
                state = np.random.get_state()
                np.random.seed(i)
                
                idxs = np.random.choice(N, size=bud, replace=False)
                train_sub_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, idxs),batch_size=train_batch_size, 
                    shuffle=False,**kwargs)# num_workers=2)
                
                np.random.set_state(state)


    # Calculate Final SubsetTrn, FullTrn, Val and Test Loss
    # Calculate Val and Test Accuracy
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        tst_correct = 0
        tst_total = 0
        for idx, test_data in  enumerate(test_loader, 0):
            inputs, target = test_data
            inputs, target = inputs.to(device), target.to(device)
            scores = model(inputs)
            test_loss = criterion(scores, target)
            total_test_loss += test_loss.item()
            _, predicted = scores.max(1)
            tst_total += target.size(0)
            tst_correct += predicted.eq(target).sum().item()
        
        tst_acc = 100.0 * tst_correct/tst_total
        tst_loss = total_test_loss/(idx+1)
        
        total_train_loss = 0
        for idx, train_data in  enumerate(train_loader, 0):
            inputs, target = train_data
            inputs, target = inputs.to(device), target.to(device)
            scores = model(inputs)
            tr_loss = criterion(scores, target)
            total_train_loss += tr_loss.item()
        print(idx)

        full_trn_loss = (1.0*total_train_loss)/(idx+1)
        
        total_valid_loss = 0
        val_correct = 0
        val_total = 0
        for idx, val_data in  enumerate(valid_loader, 0):
            inputs, target = val_data
            inputs, target = inputs.to(device), target.to(device)
            scores = model(inputs)
            val_loss = criterion(scores, target)
            total_valid_loss += val_loss.item()
            _, predicted = scores.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()
        
        val_loss = (1.0*total_valid_loss)/(idx+1)

        fulltrn_losses[-1] = full_trn_loss
        val_acc = 100.0*val_correct/val_total
        
        '''sub_total = 0 
        sub_correct = 0
        total_sub_loss = 0  
        for idx, data in  enumerate(train_sub_loader, 0):    
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            scores = model(inputs)
            sub_loss = criterion(scores, target)
            total_sub_loss += sub_loss.item()
            _, predicted = scores.max(1)
            sub_total += target.size(0)
            sub_correct += predicted.eq(target).sum().item()
            
        sub_trn_loss = (total_sub_loss/sub_total)'''    
        #sub_trn_out = model(x_trn[idxs])
            #sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
        
        

    print("SelectionRun---------------------------------")
    #print("Final SubsetTrn and FullTrn Loss:", sub_trn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", tst_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc, val_loss, tst_loss, substrn_losses, fulltrn_losses, val_losses



# list_fractions = [0.1, 0.2, 0.4, 0.75, 0.9,1]
list_fractions = [0.2,0.5,0.8]
for fraction in list_fractions:
    bud = int(fraction * N)
    print("Budget, fraction and N:", bud, fraction, N)

    # Starting Random idxs to use for Random, Taylor, and Non-Taylor version.
    start_idxs = np.random.choice(N, size=bud, replace=False)

    # Facility Location Run
    fc_valacc, fc_tstacc, fc_valloss, fc_tstloss, fc_substrn_losses, fc_fulltrn_losses, fc_val_losses= train_model(SetFunctionFacLoc,None,bud)
   
    
    ## KnnSB selection with Flag = TRN and FLAG = VAL
    knn_idxs_flag_trn = perform_knnsb_selection(datadir, data_name, fraction, selUsing='trn')
    knn_idxs_flag_val = perform_knnsb_selection(datadir, data_name, fraction, selUsing='val')

    ## Training with KnnSB idxs with Flag = TRN and FLAG = VAL
    knn_valacc_flagtrn, knn_tstacc_flagtrn, knn_valloss_flagtrn, knn_tstloss_flagtrn, knn_ftrn_substrn_losses, knn_ftrn_fulltrn_losses, knn_ftrn_val_losses = train_model("KNNSB",knn_idxs_flag_trn)
    knn_valacc_flagval, knn_tstacc_flagval, knn_valloss_flagval, knn_tstloss_flagval, knn_fval_substrn_losses, knn_fval_fulltrn_losses, knn_fval_val_losses = train_model("KNNSB",knn_idxs_flag_val)
    
    
    # OneStep Taylor Runs    
    t_trn_valacc, t_trn_tstacc, t_trn_valloss, t_trn_tstloss, tay_ftrn_substrn_losses, tay_ftrn_fulltrn_losses, tay_ftrn_val_losses = train_model(SetFunctionTaylorDeep,start_idxs,bud,False)
    t_val_valacc, t_val_tstacc, t_val_valloss, t_val_tstloss, tay_fval_substrn_losses, tay_fval_fulltrn_losses, tay_fval_val_losses = train_model(SetFunctionTaylorDeep,start_idxs,bud,True)
    
    
    #Random Run 
    rv1, rt1, rv2, rt2, rand_substrn_losses, rand_fulltrn_losses, rand_val_losses = train_model("Random",start_idxs)    

    # Random Online Selection Run
    orv1, ort1, orv2, ort2, o_rand_substrn_losses, o_rand_fulltrn_losses, o_rand_val_losses = train_model("Online-Random",start_idxs, bud)    

    # if compare_method:
    # OneStep NON-taylor Run    
    #nt_trn_valacc, nt_trn_tstacc, nt_trn_valloss, nt_trn_tstloss, nontay_ftrn_substrn_losses, nontay_ftrn_fulltrn_losses, nontay_ftrn_val_losses = train_model_online_old(start_idxs,bud,False, do_compare=compare_method)
    #nt_val_valacc, nt_val_tstacc, nt_val_valloss, nt_val_tstloss, nontay_fval_substrn_losses, nontay_fval_fulltrn_losses, nontay_fval_val_losses = train_model_online_old(start_idxs,bud,True, do_compare=compare_method)
    

    ########################################################################
    ###### Subset Trn loss with val = TRN #############
    plt.figure()
    plt.plot(np.arange(1,num_epochs+1), knn_ftrn_substrn_losses, 'r-', label='knn_v=trn')
    plt.plot(np.arange(1,num_epochs+1), tay_ftrn_substrn_losses, 'b-', label='tay_v=trn')
    plt.plot(np.arange(1,num_epochs+1), rand_substrn_losses, 'g-', label='random')
    plt.plot(np.arange(1,num_epochs+1), fc_substrn_losses, '#000000', label='Facility Location')
    #plt.plot(np.arange(1,num_epochs+1), nontay_ftrn_substrn_losses, 'orange', label='NON-tay_v=trn')
    plt.plot(np.arange(1,num_epochs+1), o_rand_substrn_losses, 'pink', label='random-online')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Subset trn loss')
    plt.title('Subset Training Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=TRN')
    plt_file = path_logfile + '_' + str(fraction) + 'substrn_loss_v=TRN.png'
    plt.savefig(plt_file)
    plt.clf()

    ###### Subset Trn loss with val = VAL #############
    plt.figure()
    plt.plot(np.arange(1,num_epochs+1), knn_fval_substrn_losses, 'r-', label='knn_v=val')
    plt.plot(np.arange(1,num_epochs+1), tay_fval_substrn_losses, 'b-', label='tay_v=val')
    plt.plot(np.arange(1,num_epochs+1), rand_substrn_losses, 'g-', label='random')
    plt.plot(np.arange(1,num_epochs+1), fc_substrn_losses, '#000000', label='Facility Location')
    #plt.plot(np.arange(1,num_epochs+1), nontay_fval_substrn_losses, 'orange', label='NON-tay_v=val')
    plt.plot(np.arange(1,num_epochs+1), o_rand_substrn_losses, 'pink', label='random-online')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Subset trn loss')
    plt.title('Subset Training Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
    plt_file = path_logfile + '_' + str(fraction) + 'substrn_loss_v=VAL.png'
    plt.savefig(plt_file)
    plt.clf()


    ########################################################################
    ###### Full Trn loss with val = TRN #############
    plt.figure()
    plt.plot(np.arange(1,num_epochs+1), knn_ftrn_fulltrn_losses[:-1], 'r-', label='knn_v=trn')
    plt.plot(np.arange(1,num_epochs+1), tay_ftrn_fulltrn_losses[:-1], 'b-', label='tay_v=trn')
    plt.plot(np.arange(1,num_epochs+1), rand_fulltrn_losses[:-1], 'g-', label='random')
    plt.plot(np.arange(1,num_epochs+1), fc_fulltrn_losses[:-1], '#000000', label='Facility Location')
    #plt.plot(np.arange(1,num_epochs+1), nontay_ftrn_fulltrn_losses[:-1], 'orange', label='NON-tay_v=trn')
    plt.plot(np.arange(1,num_epochs+1), o_rand_fulltrn_losses[:-1], 'pink', label='random-online')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Full trn loss')
    plt.title('Full Training Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=TRN')
    plt_file = path_logfile + '_' + str(fraction) + 'fulltrn_loss_v=TRN.png'
    plt.savefig(plt_file)
    plt.clf()

    ###### Full Trn loss with val = VAL #############
    plt.figure()
    plt.plot(np.arange(1,num_epochs+1), knn_fval_fulltrn_losses[:-1], 'r-', label='knn_v=val')
    plt.plot(np.arange(1,num_epochs+1), tay_fval_fulltrn_losses[:-1], 'b-', label='tay_v=val')
    plt.plot(np.arange(1,num_epochs+1), rand_fulltrn_losses[:-1], 'g-', label='random')
    plt.plot(np.arange(1,num_epochs+1), fc_fulltrn_losses[:-1], '#000000', label='Facility Location')
    #plt.plot(np.arange(1,num_epochs+1), nontay_fval_fulltrn_losses[:-1], 'orange', label='NON-tay_v=val')
    plt.plot(np.arange(1,num_epochs+1), o_rand_fulltrn_losses[:-1], 'pink', label='random-online')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Full trn loss')
    plt.title('Full Training Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
    plt_file = path_logfile + '_' + str(fraction) + 'fulltrn_loss_v=VAL.png'
    plt.savefig(plt_file)
    plt.clf()


    ########################################################################
    ###### Validation loss with val = TRN #############
    plt.figure()
    plt.plot(np.arange(1,num_epochs+1), knn_ftrn_val_losses, 'r-', label='knn_v=trn')
    plt.plot(np.arange(1,num_epochs+1), tay_ftrn_val_losses, 'b-', label='tay_v=trn')
    plt.plot(np.arange(1,num_epochs+1), rand_val_losses, 'g-', label='random')
    plt.plot(np.arange(1,num_epochs+1), fc_val_losses, '#000000', label='Facility Location')
    #plt.plot(np.arange(1,num_epochs+1), nontay_ftrn_val_losses, 'orange', label='NON-tay_v=trn')
    plt.plot(np.arange(1,num_epochs+1), o_rand_val_losses, 'pink', label='random-online')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Validation loss')
    plt.title('Validation Loss vs Epochs ' + data_name + '_' + str(fraction)+ '_' + 'val=TRN')
    plt_file = path_logfile + '_' + str(fraction) + 'valloss_v=TRN.png'
    plt.savefig(plt_file)
    plt.clf()

    ###### Validation loss with val = VAL #############
    plt.figure()
    plt.plot(np.arange(1,num_epochs+1), knn_fval_val_losses, 'r-', label='knn_v=val')
    plt.plot(np.arange(1,num_epochs+1), tay_fval_val_losses, 'b-', label='tay_v=val')
    plt.plot(np.arange(1,num_epochs+1), rand_val_losses, 'g-', label='random')
    plt.plot(np.arange(1,num_epochs+1), fc_val_losses, '#000000', label='Facility Location')
    #plt.plot(np.arange(1,num_epochs+1), nontay_fval_val_losses, 'orange', label='NON-tay_v=val')
    plt.plot(np.arange(1,num_epochs+1), o_rand_val_losses, 'pink', label='random-online')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Validation loss')
    plt.title('Validation Loss vs Epochs ' + data_name + '_' + str(fraction)+ '_' + 'val=VAL')
    plt_file = path_logfile + '_' + str(fraction) + 'valloss_v=VAL.png'
    plt.savefig(plt_file)
    plt.clf()

    print(data_name,":Budget = ",fraction,file=logfile)
    print('---------------------------------------------------------------------',file=logfile)
    print('|Algo                            | Val Acc       |   Test Acc       |',file=logfile)
    print('| -------------------------------|:-------------:| ----------------:|',file=logfile)
    print('*| Facility Location             |',fc_valacc, '  | ',fc_tstacc,' |',file=logfile)
    print('*| Taylor with Validation=TRN     |', t_trn_valacc , '  | ', t_trn_tstacc ,' |',file=logfile)
    print('*| Taylor with Validation=VAL     |', t_val_valacc , '  | ', t_val_tstacc ,' |',file=logfile)
    print('*| KnnSB with Validation=TRN      |', knn_valacc_flagtrn, '| ',knn_tstacc_flagtrn,' |',file=logfile)
    print('*| KnnSB with Validation=VAL      |', knn_valacc_flagval, '| ',knn_tstacc_flagval,' |',file=logfile)
    print('*| Random Selection               |', rv1,              '  | ', rt1,              ' |',file=logfile)
    print('*| Online Random Selection        |', orv1,              '  | ', ort1,              ' |',file=logfile)
    #print('*| NON-Tay with Validation=TRN    |', nt_trn_valacc , '  | ', nt_trn_tstacc ,' |',file=logfile)
    #print('*| NON-Tay with Validation=VAL    |', nt_val_valacc , '  | ', nt_val_tstacc ,' |',file=logfile)
    
    print("\n", file=logfile)

    print("=========Facility Location===================", file=logfile)       
    print("*FacLoc Validation LOSS:", fc_valloss, file=logfile)    
    print("*FacLoc Test Data LOSS:", fc_tstloss,file=logfile)
    print("*FacLoc Full Trn Data LOSS:", fc_fulltrn_losses[-1],file=logfile)
       
    print("=========Online Selection Taylor without Validation Set===================", file=logfile)      
    print("*Taylor v=TRN Validation LOSS:", t_trn_valloss, file=logfile)
    print("*Taylor v=TRN Test Data LOSS:", t_trn_tstloss, file=logfile)
    print("*Taylor v=TRN Full Trn Data LOSS:", tay_ftrn_fulltrn_losses[-1], file=logfile)

    print("=========Online Selection Taylor with Validation Set===================", file=logfile)      
    print("*Taylor v=VAL Validation LOSS:", t_val_valloss, file=logfile)
    print("*Taylor v=VAL Test Data LOSS:", t_val_tstloss, file=logfile)
    print("*Taylor v=VAL Full Trn Data LOSS:", tay_fval_fulltrn_losses[-1], file=logfile)

    print("=========kNNsb Selection with Validation = TRN ===================", file=logfile)      
    print("*KnnSB v=TRN Validation LOSS:", knn_valloss_flagtrn, file=logfile)
    print("*KnnSB v=TRN Test Data LOSS:", knn_tstloss_flagtrn, file=logfile)
    print("*KnnSB v=TRN Full Trn LOSS:", knn_ftrn_fulltrn_losses[-1], file=logfile)

    print("=========kNNsb Selection with Validation = VAL ===================", file=logfile)      
    print("*KnnSB v=VAL Validation LOSS:", knn_valloss_flagval, file=logfile)
    print("*KnnSB v=VAL Test Data LOSS:", knn_tstloss_flagval, file=logfile)
    print("*KnnSB v=VAL Full Trn Data LOSS:", knn_fval_fulltrn_losses[-1], file=logfile)

    print("=========Random Results==============", file=logfile)       
    print("*Rand Validation LOSS:", rv2, file=logfile)
    print("*Rand Test Data LOSS:", rt2, file=logfile)
    print("*Rand Full Trn Data LOSS:", rand_fulltrn_losses[-1], file=logfile)

    print("=========Online Random Results==============", file=logfile)       
    print("*Rand Validation LOSS:", orv2, file=logfile)
    print("*Rand Test Data LOSS:", ort2, file=logfile)
    print("*Rand Full Trn Data LOSS:", rand_fulltrn_losses[-1], file=logfile)

    '''print("=========Online Selection NON-taylor without Validation Set===================", file=logfile)      
    print("*Taylor v=TRN Validation LOSS:", t_trn_valloss, file=logfile)
    print("*Taylor v=TRN Test Data LOSS:", t_trn_tstloss, file=logfile)
    print("*Taylor v=TRN Full Trn Data LOSS:", tay_ftrn_fulltrn_losses[-1], file=logfile)

    print("=========Online Selection NON-taylor with Validation Set===================", file=logfile)      
    print("*Taylor v=VAL Validation LOSS:", nt_val_valloss, file=logfile)
    print("*Taylor v=VAL Test Data LOSS:", nt_val_tstloss, file=logfile)
    print("*Taylor v=VAL Full Trn Data LOSS:", nontay_fval_fulltrn_losses[-1], file=logfile)'''

    print("=============================================================================================", file=logfile) 
    print("---------------------------------------------------------------------------------------------", file=logfile)
    print("\n", file=logfile)       
