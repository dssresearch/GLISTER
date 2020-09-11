import sys
import datetime
import numpy as np
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.logistic_regression import LogisticRegNet
#from models.simpleNN_net import ThreeLayerNet
#from sklearn.model_selection import train_test_split
from utils.data_utils import load_dataset_numpy, write_knndata


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using Device:", device)

## Convert to this argparse 
datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
rnd_seed = int(sys.argv[5])
sel_using = sys.argv[6] # == 'val' or 'trn'. Which dataset to use for selection

learning_rate = 0.1
torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed)

run_path = './run_data/'  # Run path of knnsubmod -- indices stored in this folder
output_dir = run_path + 'KNNSubmod_' + data_name + '/'
indices_file = output_dir + 'KNNSubmod_' + str((int)(fraction*100)) + ".subset"

path_logfile = './data/' + data_name + '_knnsb.txt'
logfile = open(path_logfile, 'a')

exp_name = data_name + '_KnnSB_fraction:' + str(fraction) + '_epochs:' + \
    str(num_epochs) + '_seed:' + str(rnd_seed)
exp_start_time = datetime.datetime.now()

print("------------------------------------", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)

# Perform Selection
# Will call the executable KNNSubmod
perform_knnsb_selection(datadir, data_name, fraction, sel_using)

fullset, valset, testset, num_cls = load_dataset_numpy(datadir, data_name)
x_trn, y_trn = fullset
x_val, y_val = valset
x_tst, y_tst = testset
# Load as a Pytorch Tensor
x_trn = torch.from_numpy(x_trn.astype(np.float32))
x_val = torch.from_numpy(x_val.astype(np.float32))
x_tst = torch.from_numpy(x_tst.astype(np.float32))
y_trn = torch.from_numpy(y_trn.astype(np.int64))
y_val = torch.from_numpy(y_val.astype(np.int64))
y_tst = torch.from_numpy(y_tst.astype(np.int64))

# Get validation data: Its 10% of the entire (full) training data
#x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1)

print('-----------------------------------------')
print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
#print(y_trn.shape, y_val.shape, y_tst.shape)

N, M = x_trn.shape
model = LogisticRegNet(M, num_cls)
#model = ThreeLayerNet(M, num_cls, 5, 5)
# if data_name == 'mnist':
#     model = MnistNet()
# if torch.cuda.device_count() > 1:
#    print("Using:", torch.cuda.device_count(), "GPUs!")
#    model = nn.DataParallel(model)
#    cudnn.benchmark = True
model = model.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

bud = int(fraction * N)
# Load the indices from the KNNSubmod output file!
idxs = np.genfromtxt(indices_file, delimiter=',', dtype=int) # since they are indices!
# idxs = np.random.randint(N, size=bud)
print("Budget, fraction and N:", bud, fraction, N)
# Transfer all the data to GPU
d_t = time.time()
x_trn, y_trn = x_trn.to(device), y_trn.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)
print("Transferred data to device in time:", time.time()-d_t)

print_every = 10
for i in range(1, num_epochs+1):    
    # inputs, targets = x_trn[idxs].to(device), y_trn[idxs].to(device)
    inputs, targets = x_trn[idxs], y_trn[idxs]
    optimizer.zero_grad()
    scores = model(inputs)
    loss = criterion(scores, targets).mean()    
    loss.backward()
    optimizer.step()

    if i % print_every == 0:  # Print Training and Validation Loss
        with torch.no_grad():
            # val_in, val_t = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val).mean()
            full_trn_outputs = model(x_trn)
            full_trn_loss = criterion(full_trn_outputs, y_trn).mean()        
        print('Epoch:', i, 'SubsetTrn,FullTrn,ValLoss:', loss.item(), full_trn_loss.item(), val_loss.item(), file=logfile)


# Calculate Final SubsetTrn, FullTrn, Val and Test Loss
# Calculate Val and Test Accuracy
model.eval()
with torch.no_grad():
    full_trn_out = model(x_trn)
    full_trn_loss = criterion(full_trn_out, y_trn).mean()
    sub_trn_out = model(x_trn[idxs])
    sub_trn_loss = criterion(sub_trn_out, y_trn[idxs]).mean()
    val_out = model(x_val)
    val_loss = criterion(val_out, y_val).mean()
    _, val_predict = val_out.max(1)
    val_correct = val_predict.eq(y_val).sum().item()
    val_total = y_val.size(0)

correct = 0
total = 0
with torch.no_grad():
    inputs, targets = x_tst.to(device), y_tst.to(device)
    outputs = model(inputs)
    test_loss = criterion(outputs, targets).mean()    
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
tst_acc = 100.0 * correct/total

print("---------------------------------", file=logfile)
print("Final SubsetTrn and FullTrn Loss:", full_trn_loss.item(), sub_trn_loss.item(), file=logfile)
print("Validation Loss and Accuracy:", val_loss.item(), 100*val_correct/val_total, file=logfile)
print("Test Data Loss and Accuracy:", test_loss.item(), tst_acc, file=logfile)
print("Test accuracy: ", tst_acc, file=logfile)
exp_end_time = datetime.datetime.now()
print("Experiment run ended at:", str(exp_end_time), file=logfile)
print("===================================", file=logfile)
logfile.close()
print('-----------------------------------')

#
#model_path = './data/mnist_model_israndom' + str(random_method) + '.pth'
#dict_path = './data/mnist_dict_israndom' + str(random_method) + '.pkl'
#torch.save(model.state_dict(), model_path)
#
#result_dict = {}
#result_dict['acc'] = acc
#result_dict['model'] = model.state_dict()
#with open(dict_path, 'wb') as ofile:
#    pickle.dump(result_dict, ofile)



