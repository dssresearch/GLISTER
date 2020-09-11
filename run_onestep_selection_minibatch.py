import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.logistic_regression import LogisticRegNet
from models.mnist_net import MnistNet
# from models.set_function_onestep import SetFunctionLoader as SetFunction
from models.set_function_onestep import SetFunctionLoader_2 as SetFunction
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from utils.data_utils import load_dataset_pytorch

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using Device:", device)

## Convert to this argparse 
datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
select_every = int(sys.argv[5])
random_method = int(sys.argv[6])
rnd_seed = int(sys.argv[7])

learning_rate = 0.1

torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed)

path_logfile = './data/' + data_name + '_loader.txt'
logfile = open(path_logfile, 'a')
exp_name = data_name + 'loader_frac:' + str(fraction) + '_epc:' + str(num_epochs) +  \
    '_sel:' + str(select_every) + '_isRand:' + str(random_method) + '_seed:' + str(rnd_seed)
exp_start_time = datetime.datetime.now()
print("------------------------------------", file=logfile)
print(exp_name, str(exp_start_time), '\n', file=logfile)


fullset, testset, num_cls = load_dataset_pytorch(datadir, data_name)
# Validation Data set is 10% of the Entire Trainset.
validation_set_fraction = 0.1
num_fulltrn = len(fullset)
num_val = int(num_fulltrn * validation_set_fraction)
num_trn = num_fulltrn - num_val
trainset, validset = random_split(fullset, [num_trn, num_val])

# all_trn_idxs = np.arange(num_fulltrn)
# split = int(np.floor(validation_set_fraction * num_fulltrn))
# train_idxs = all_trn_idxs[split:]
# val_idxs = all_trn_idxs[:split]
# trn_sampler = SubsetRandomSampler(train_idxs) 
# val_sampler = SubsetRandomSampler(val_idxs)

# curr_batch_size = 256
trn_batch_size = 20
val_batch_size = 20
tst_batch_size = 20


trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, 
    shuffle=False, num_workers=1, pin_memory=True)

valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size, 
    shuffle=False, num_workers=1, pin_memory=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size, 
    shuffle=False, num_workers=1, pin_memory=True)


# Utility to give out the evaluation loss
def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss



N = len(trainset)
M = trainset.dataset.data.shape[1]
print("trainset size:", N ,M)

model = LogisticRegNet(M, num_cls)
if data_name == 'mnist':
    model = MnistNet()
# if torch.cuda.device_count() > 1:
#    print("Using:", torch.cuda.device_count(), "GPUs!")
#    model = nn.DataParallel(model)
#    cudnn.benchmark = True
model = model.to(device)

criterion = nn.CrossEntropyLoss() #defaults to reduction='mean'
criterion_nored = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)
subset_idxs = np.random.randint(N, size=bud)
subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, 
    shuffle=False, sampler=SubsetRandomSampler(subset_idxs), num_workers=1, pin_memory=True)

# setf_model = SetFunction(trainloader, valloader, model, criterion, 
#     criterion_nored, learning_rate, device)

## Pass in the entire validation data set!
setf_model = SetFunction(trainloader, validset, model, criterion, 
    criterion_nored, learning_rate, device)


print_every = 2
for i in range(1, num_epochs+1):
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

    if i % print_every == 0:
        t=time.time()
        val_loss = model_eval_loss(valloader, model, criterion)
        print("Val loss comp time:", time.time()-t)
        alltrn_loss = model_eval_loss(trainloader, model, criterion)
        print('Epoch:', i, 'SubsetTrn,allTrn,ValLoss:', subtrn_loss, alltrn_loss, val_loss, file=logfile)

    if (not random_method) and ((i+1) % select_every == 0):
        cached_state_dict = copy.deepcopy(model.state_dict())
        clone_dict = copy.deepcopy(model.state_dict())
        print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))        
        subset_idxs, grads_idxs = setf_model.naive_greedy_max(bud, clone_dict)
        print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
        model.load_state_dict(cached_state_dict)
        ### Change the subset_trnloader according to new found indices: subset_idxs 
        subset_trnloader = torch.utils.data.DataLoader(fullset, batch_size=trn_batch_size, shuffle=False, 
            sampler=SubsetRandomSampler(subset_idxs), num_workers=1, pin_memory=True)

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
val_acc = val_correct/val_total

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
tst_acc = correct/total
print("Validation Loss and Accuracy:", val_loss, val_acc, file=logfile)
print("Test Data Loss and Accuracy:", test_loss, tst_acc, file=logfile)
exp_end_time = datetime.datetime.now()
print("Experiment run ended at:", str(exp_end_time), file=logfile)
print("===================================", file=logfile)
logfile.close()
print('-----------------------------------')


