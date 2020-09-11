import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn import datasets
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler,  MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from matplotlib import pyplot as plt

## Custom PyTorch Dataset Class wrapper
class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None):       
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = torch.from_numpy(data.astype('float32'))#.to(device)
            self.targets = torch.from_numpy(target)#.to(device)
        else:
            self.data = data.astype('float32')
            self.targets = target

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return (sample_data, label) #.astype('float32')



## Utility function to load datasets from libsvm datasets
def libsvm_file_load(path,dim, save_data=False):
    data = []
    target = []
    with open(path) as fp:
       line = fp.readline()
       while line:
        temp = [i for i in line.strip().split(" ")]
        target.append(int(float(temp[0]))) # Class Number. # Not assumed to be in (0, K-1)
        temp_data = [0]*dim
        
        for i in temp[1:]:
            ind,val = i.split(':')
            temp_data[int(ind)-1] = float(val)
        data.append(temp_data)
        line = fp.readline()
    X_data = np.array(data,dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)


## Utility function to load datasets from libsvm datasets
## path = input file path
def libsvm_to_standard(path, dim):
    data = []
    target = []
    with open(path) as fp:
       line = fp.readline()
       while line:
        temp = [i for i in line.strip().split(" ")]
        row_vector = [0] * (dim+1) # +1 for the y label
        row_label = int(temp[0])
        target.append(row_label) # Class Number. # Not assumed to be in (0, K-1)
        # row_vector[0] = row_label
        temp_data = [0] * dim        
        for i in temp[1:]:
            ind,val = i.split(':')
            temp_data[int(ind)-1] = float(val)
        # insert class label as the 1st column
        temp_data.insert(0, row_label)
        data.append(temp_data)
        line = fp.readline()
    
    all_data = np.array(data,dtype=np.float32)
    np.savetxt(path + ".trf", all_data, fmt='%.3f')   # entire data
    # return all_data




## Function to load a discrete UCI dataset and make it ordinal.
def clean_uci_ordinal_data(inp_fname, out_fname):
    # trn, val, tst split: 0.8, 0.1, 0.1
    data = np.genfromtxt(inp_fname, delimiter=',', dtype='str')
    enc = OrdinalEncoder(dtype=int)
    enc.fit(data)
    transformed_data = enc.transform(data)
    np.random.shuffle(transformed_data)     # randomly shuffle the data points
    
    N = transformed_data.shape[0]
    N_trn = int(N * 0.8)
    N_val = int(N * 0.1)
    N_tst = int(N * 0.1)
    data_trn = transformed_data[: N_trn]
    data_val = transformed_data[N_trn : N_trn + N_val]
    data_tst = transformed_data[N_trn + N_val :]

    np.savetxt(out_fname + ".full", transformed_data, fmt='%.0f')   # entire data
    np.savetxt(out_fname + ".trn", data_trn, fmt='%.0f')
    np.savetxt(out_fname + ".val", data_val, fmt='%.0f')
    np.savetxt(out_fname + ".tst", data_tst, fmt='%.0f')



## Utility function to save numpy array for knnSB
## Used in: KnnSubmod Selection
def write_knndata(datadir, dset_name,feature):

    # Write out the trndata
    trn_filepath = os.path.join(datadir, feature+'_knn_' + dset_name + '.trn')
    val_filepath = os.path.join(datadir, feature+'_knn_' + dset_name + '.val')
    tst_filepath = os.path.join(datadir, feature+'_knn_' + dset_name + '.tst')

    if os.path.exists(trn_filepath) and os.path.exists(val_filepath) and os.path.exists(tst_filepath):
        return

    if dset_name in ['mnist', "fashion-mnist"] :
        fullset, testset, num_cls  = load_dataset_numpy(datadir, dset_name,feature=feature)
        x_trn, y_trn = fullset.data, fullset.targets
        x_tst, y_tst = testset.data, testset.targets
        x_trn = x_trn.view(x_trn.shape[0], -1)
        x_tst = x_tst.view(x_tst.shape[0], -1)
        # Get validation data: Its 10% of the entire (full) training data
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

    else:
        fullset, valset, testset, num_cls  = load_dataset_numpy(datadir, dset_name, feature)
    
        x_trn, y_trn = fullset
        x_val , y_val = valset
        x_tst, y_tst = testset
    ## Create VAL data
    #x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

    trndata = np.c_[x_trn, y_trn]
    valdata = np.c_[x_val, y_val]
    tstdata = np.c_[x_tst, y_tst]


    np.savetxt(trn_filepath, trndata, fmt='%.6f')
    np.savetxt(val_filepath, valdata, fmt='%.6f')
    np.savetxt(tst_filepath, tstdata, fmt='%.6f')

    return

## Takes in a dataset name and returns a PyTorch Dataset Object
def load_dataset_pytorch(datadir, dset_name,device):
    if dset_name == "dna":
        np.random.seed(42)
        trn_file = os.path.join(datadir, 'dna.scale.trn')
        val_file = os.path.join(datadir, 'dna.scale.val')
        tst_file = os.path.join(datadir, 'dna.scale.tst')
        data_dims = 180
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        #x_trn = np.concatenate((x_trn, x_val))
        #y_trn = np.concatenate((y_trn, y_val))
        y_trn -= 1  # First Class should be zero
        y_val -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)
        return fullset, valset, testset, data_dims,num_cls 

    elif dset_name == "sensit_seismic":
        trn_file = os.path.join(datadir, 'sensit_seismic.trn')
        tst_file = os.path.join(datadir, 'sensit_seismic.tst')
        data_dims = 50
        num_cls = 3
        
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)
        return fullset, valset, testset, data_dims,num_cls 

    elif dset_name == "protein":
        trn_file = os.path.join(datadir, 'protein.trn')
        val_file = os.path.join(datadir, 'protein.val')
        tst_file = os.path.join(datadir, 'protein.tst')
        data_dims = 357
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        #x_trn = np.concatenate((x_trn, x_val))
        #y_trn = np.concatenate((y_trn, y_val))
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)
        return fullset, valset, testset, data_dims,num_cls

    elif dset_name == "shuttle":
        trn_file = os.path.join(datadir, 'shuttle.trn')
        val_file = os.path.join(datadir, 'shuttle.val')
        tst_file = os.path.join(datadir, 'shuttle.tst')
        data_dims = 9
        num_cls = 7
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        x_trn = np.concatenate((x_trn, x_val))
        y_trn = np.concatenate((y_trn, y_val))
        y_trn -= 1  # First Class should be zero
        #y_val -= 1
        y_tst -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)

        return fullset, valset, testset, data_dims,num_cls

    elif dset_name == "sensorless":
        trn_file = os.path.join(datadir, 'sensorless.scale.trn')
        tst_file = os.path.join(datadir, 'sensorless.scale.val')
        data_dims = 48
        num_cls = 11
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)

        return fullset, valset, testset, data_dims, num_cls

    elif dset_name == "connect_4":
        trn_file = os.path.join(datadir, 'connect_4.trn')

        data_dims = 126
        num_cls = 3

        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        # The class labels are (-1,0,1). Make them to (0,1,2)
        y_trn[y_trn < 0] = 2

        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)

        return fullset, valset, testset, data_dims,num_cls


    elif dset_name == "letter":
        trn_file = os.path.join(datadir, 'letter.scale.trn')
        val_file = os.path.join(datadir, 'letter.scale.val')
        tst_file = os.path.join(datadir, 'letter.scale.tst')
        data_dims = 16
        num_cls = 26 
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        #x_trn = np.concatenate((x_trn, x_val))
        #y_trn = np.concatenate((y_trn, y_val))

        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)
        
        return fullset, valset, testset, data_dims,num_cls
   
    elif dset_name == "pendigits":
        trn_file = os.path.join(datadir, 'pendigits.trn_full')
        tst_file = os.path.join(datadir, 'pendigits.tst')
        data_dims = 16
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

        sc = StandardScaler()
        
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)
        
        return fullset, valset, testset, data_dims,num_cls
    

    elif dset_name == "satimage":
        trn_file = os.path.join(datadir, 'satimage.scale.trn')
        val_file = os.path.join(datadir, 'satimage.scale.val')
        tst_file = os.path.join(datadir, 'satimage.scale.tst')
        data_dims = 36
        num_cls = 6
        
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        #x_trn = np.concatenate((x_trn, x_val))
        #y_trn = np.concatenate((y_trn, y_val))
        
        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)
        
        return fullset, valset, testset, data_dims,num_cls

    elif dset_name == "svmguide1":
        trn_file = os.path.join(datadir, 'svmguide1.trn_full')
        tst_file = os.path.join(datadir, 'svmguide1.tst')
        data_dims = 4
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst, device)
        return fullset, valset, testset, data_dims,num_cls
    
    elif dset_name == "usps":
        trn_file = os.path.join(datadir, 'usps.trn_full')
        tst_file = os.path.join(datadir, 'usps.tst')
        data_dims = 256
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst, device)
        return fullset, valset, testset, data_dims,num_cls
    
    elif dset_name == "ijcnn1":
        trn_file = os.path.join(datadir, 'ijcnn1.trn')
        val_file = os.path.join(datadir, 'ijcnn1.val')
        tst_file = os.path.join(datadir, 'ijcnn1.tst')
        data_dims = 22
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        #x_trn = np.concatenate((x_trn, x_val))
        #y_trn = np.concatenate((y_trn, y_val))
        
        # The class labels are (-1,1). Make them to (0,1)
        y_trn[y_trn < 0] = 0
        y_val[y_val < 0] = 0
        y_tst[y_tst < 0] = 0
       
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        
        fullset = CustomDataset(x_trn, y_trn,device)
        valset = CustomDataset(x_val, y_val,device)
        testset = CustomDataset(x_tst, y_tst,device)
        
        return fullset, valset, testset, data_dims,num_cls
    elif dset_name == "mnist":
        mnist_transform = transforms.Compose([
            transforms.Grayscale(3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ])
        num_cls = 10
        fullset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=mnist_transform)
        testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=mnist_transform)
        return fullset, testset, num_cls
    elif dset_name == "sklearn-digits":
        data, target = datasets.load_digits(return_X_y=True)
        # Test data is 10%
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        num_cls = 10
        fullset = CustomDataset(x_trn, y_trn, device)
        valset = CustomDataset(x_val, y_val, device)
        testset = CustomDataset(x_tst, y_tst, device)
        return fullset, valset, testset, x_trn.shape[1], num_cls
    elif dset_name == "bc":
        data, target = datasets.load_breast_cancer(return_X_y=True)
        
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        num_cls = 2
        
        fullset = CustomDataset(x_trn, y_trn, device)
        valset = CustomDataset(x_val, y_val, device)
        testset = CustomDataset(x_tst, y_tst, device)
        return fullset, valset, testset, x_trn.shape[1], num_cls


## Takes in a dataset name and returns a Tuple of ((x_trn, y_trn), (x_tst, y_tst))
def load_dataset_numpy(datadir, dset_name, feature=None):
    if dset_name == "dna":
        trn_file = os.path.join(datadir, 'dna.scale.trn')
        val_file = os.path.join(datadir, 'dna.scale.val')
        tst_file = os.path.join(datadir, 'dna.scale.tst')
        data_dims = 180
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        #x_trn = np.concatenate((x_trn, x_val))
        #y_trn = np.concatenate((y_trn, y_val))
        y_trn -= 1  # First Class should be zero
        y_val -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        if feature == 'classimb':
            samples_per_class = np.zeros(num_cls)
            val_samples_per_class = np.zeros(num_cls)
            tst_samples_per_class = np.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(np.where(y_trn == i)[0])
                val_samples_per_class[i] = len(np.where(y_val == i)[0])
                tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
            min_samples = int(np.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = x_trn[subset_idxs]
                    y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
                else:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
                    y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))
            min_samples = int(np.min(val_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = x_val[subset_ids]
                    y_val_new = y_val[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = np.row_stack((x_val_new, x_val[subset_ids]))
                    y_val_new = np.row_stack((y_val_new, y_val[subset_ids].reshape(-1, 1)))
            min_samples = int(np.min(tst_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = x_tst[subset_ids]
                    y_tst_new = y_tst[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = np.row_stack((x_tst_new, x_tst[subset_ids]))
                    y_tst_new = np.row_stack((y_tst_new, y_tst[subset_ids].reshape(-1, 1)))
            x_val = x_val_new
            y_val = y_val_new
            x_tst = x_tst_new
            y_tst = y_tst_new
            y_trn = y_trn_new
            x_trn = x_trn_new
        elif feature == 'noise':
            noise_size = int(len(y_trn) * 0.8)
            noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
            y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)

        fullset = (x_trn, y_trn.reshape(-1))
        valset = (x_val, y_val.reshape(-1))
        testset = (x_tst, y_tst.reshape(-1))
        return fullset, valset, testset, num_cls 

    elif dset_name == "sensit_seismic":
        trn_file = os.path.join(datadir, 'sensit_seismic.trn')
        tst_file = os.path.join(datadir, 'sensit_seismic.tst')
        data_dims = 50
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)
        return fullset, valset, testset, num_cls

    elif dset_name == "protein":
        trn_file = os.path.join(datadir, 'protein.trn')
        val_file = os.path.join(datadir, 'protein.val')
        tst_file = os.path.join(datadir, 'protein.tst')
        data_dims = 357
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        #x_trn = np.concatenate((x_trn, x_val))
        #y_trn = np.concatenate((y_trn, y_val))
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)
        return fullset, valset, testset, num_cls

    elif dset_name == "shuttle":
        trn_file = os.path.join(datadir, 'shuttle.trn')
        val_file = os.path.join(datadir, 'shuttle.val')
        tst_file = os.path.join(datadir, 'shuttle.tst')
        data_dims = 9
        num_cls = 7
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        #x_trn = np.concatenate((x_trn, x_val))
        #y_trn = np.concatenate((y_trn, y_val))
        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero

        #x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        samples_per_class = np.zeros(num_cls)
        val_samples_per_class = np.zeros(num_cls)
        tst_samples_per_class = np.zeros(num_cls)
        for i in range(num_cls):
            samples_per_class[i] = len(np.where(y_trn == i)[0])
            val_samples_per_class[i] = len(np.where(y_val == i)[0])
            tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
        min_samples = int(np.min(samples_per_class) * 0.1)

        if feature == 'classimb':
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = x_trn[subset_idxs]
                    y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
                else:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
                    y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))

        elif feature == 'noise':
            noise_size = int(len(y_trn) * 0.8)
            noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
            y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)

    
        if feature == 'classimb':
            x_val = x_val_new
            y_val = y_val_new
            x_tst = x_tst_new
            y_tst = y_tst_new
            y_trn = y_trn_new
            x_trn = x_trn_new

        fullset = (x_trn, y_trn.reshape(-1))
        valset = (x_val, y_val.reshape(-1))
        testset = (x_tst, y_tst.reshape(-1))
        return fullset, valset, testset, num_cls

    elif dset_name == "sensorless":
        trn_file = os.path.join(datadir, 'sensorless.scale.trn')
        tst_file = os.path.join(datadir, 'sensorless.scale.val')
        data_dims = 48
        num_cls = 11
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)
        return fullset, valset, testset, num_cls

    elif dset_name == "connect_4":
        trn_file = os.path.join(datadir, 'connect_4.trn')

        data_dims = 126
        num_cls = 3

        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        # The class labels are (-1,0,1). Make them to (0,1,2)
        y_trn[y_trn < 0] = 2

        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        samples_per_class = np.zeros(num_cls)
        val_samples_per_class = np.zeros(num_cls)
        tst_samples_per_class = np.zeros(num_cls)
        for i in range(num_cls):
            samples_per_class[i] = len(np.where(y_trn == i)[0])
            val_samples_per_class[i] = len(np.where(y_val == i)[0])
            tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
        min_samples = int(np.min(samples_per_class) * 0.1)

        if feature == 'classimb':
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = x_trn[subset_idxs]
                    y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
                else:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
                    y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))

            min_samples = int(np.min(val_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = x_val[subset_ids]
                    y_val_new = y_val[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = np.row_stack((x_val_new, x_val[subset_ids]))
                    y_val_new = np.row_stack((y_val_new, y_val[subset_ids].reshape(-1, 1)))
            min_samples = int(np.min(tst_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = x_tst[subset_ids]
                    y_tst_new = y_tst[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = np.row_stack((x_tst_new, x_tst[subset_ids]))
                    y_tst_new = np.row_stack((y_tst_new, y_tst[subset_ids].reshape(-1, 1)))

            x_val = x_val_new
            y_val = y_val_new
            x_tst = x_tst_new
            y_tst = y_tst_new
            y_trn = y_trn_new
            x_trn = x_trn_new


        elif feature == 'noise':
            noise_size = int(len(y_trn) * 0.8)
            noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
            y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)

        fullset = (x_trn, y_trn.reshape(-1))
        valset = (x_val, y_val.reshape(-1))
        testset = (x_tst, y_tst.reshape(-1))
        return fullset, valset, testset, num_cls

    elif dset_name == "letter":
        trn_file = os.path.join(datadir, 'letter.scale.trn')
        val_file = os.path.join(datadir, 'letter.scale.val')
        tst_file = os.path.join(datadir, 'letter.scale.tst')
        data_dims = 16
        num_cls = 26 
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        #fullset = (x_trn, y_trn)
        #valset = (x_val, y_val)
        #testset = (x_tst, y_tst)

        if feature == 'classimb':
            samples_per_class = np.zeros(num_cls)
            val_samples_per_class = np.zeros(num_cls)
            tst_samples_per_class = np.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(np.where(y_trn == i)[0])
                val_samples_per_class[i] = len(np.where(y_val == i)[0])
                tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
            min_samples = int(np.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = x_trn[subset_idxs]
                    y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
                else:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
                    y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))
            min_samples = int(np.min(val_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = x_val[subset_ids]
                    y_val_new = y_val[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = np.row_stack((x_val_new, x_val[subset_ids]))
                    y_val_new = np.row_stack((y_val_new, y_val[subset_ids].reshape(-1, 1)))
            min_samples = int(np.min(tst_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = x_tst[subset_ids]
                    y_tst_new = y_tst[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = np.row_stack((x_tst_new, x_tst[subset_ids]))
                    y_tst_new = np.row_stack((y_tst_new, y_tst[subset_ids].reshape(-1, 1)))
            x_val = x_val_new
            y_val = y_val_new
            x_tst = x_tst_new
            y_tst = y_tst_new
            y_trn = y_trn_new
            x_trn = x_trn_new
        elif feature == 'noise':
            noise_size = int(len(y_trn) * 0.8)
            noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
            y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)
        fullset = (x_trn, y_trn.reshape(-1))
        valset = (x_val, y_val.reshape(-1))
        testset = (x_tst, y_tst.reshape(-1))


        return fullset, valset, testset, num_cls

    elif dset_name == "pendigits":
        trn_file = os.path.join(datadir, 'pendigits.trn_full')
        tst_file = os.path.join(datadir, 'pendigits.tst')
        data_dims = 16
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

        sc = StandardScaler()
        
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)
        return fullset, valset, testset, num_cls

    elif dset_name == "satimage":
        np.random.seed(42)
        trn_file = os.path.join(datadir, 'satimage.scale.trn')
        val_file = os.path.join(datadir, 'satimage.scale.val')
        tst_file = os.path.join(datadir, 'satimage.scale.tst')
        data_dims = 36
        num_cls = 6
        
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        #x_trn = np.concatenate((x_trn, x_val))
        #y_trn = np.concatenate((y_trn, y_val))
        
        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        if feature == 'classimb':
            samples_per_class = np.zeros(num_cls)
            val_samples_per_class = np.zeros(num_cls)
            tst_samples_per_class = np.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(np.where(y_trn == i)[0])
                val_samples_per_class[i] = len(np.where(y_val == i)[0])
                tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
            min_samples = int(np.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = x_trn[subset_idxs]
                    y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
                else:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
                    y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))
            min_samples = int(np.min(val_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = x_val[subset_ids]
                    y_val_new = y_val[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = np.row_stack((x_val_new, x_val[subset_ids]))
                    y_val_new = np.row_stack((y_val_new, y_val[subset_ids].reshape(-1, 1)))
            min_samples = int(np.min(tst_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = x_tst[subset_ids]
                    y_tst_new = y_tst[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = np.row_stack((x_tst_new, x_tst[subset_ids]))
                    y_tst_new = np.row_stack((y_tst_new, y_tst[subset_ids].reshape(-1, 1)))
            x_val = x_val_new
            y_val = y_val_new
            x_tst = x_tst_new
            y_tst = y_tst_new
            y_trn = y_trn_new
            x_trn = x_trn_new
        elif feature == 'noise':
            noise_size = int(len(y_trn) * 0.8)
            noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
            y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)
        fullset = (x_trn, y_trn.reshape(-1))
        valset = (x_val, y_val.reshape(-1))
        testset = (x_tst, y_tst.reshape(-1))
        return fullset, valset, testset, num_cls

    elif dset_name == "svmguide1":
        np.random.seed(42)
        trn_file = os.path.join(datadir, 'svmguide1.trn_full')
        tst_file = os.path.join(datadir, 'svmguide1.tst')
        data_dims = 4
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        samples_per_class = np.zeros(num_cls)
        val_samples_per_class = np.zeros(num_cls)
        tst_samples_per_class = np.zeros(num_cls)
        for i in range(num_cls):
            samples_per_class[i] = len(np.where(y_trn == i)[0])
            val_samples_per_class[i] = len(np.where(y_val == i)[0])
            tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
        min_samples = int(np.min(samples_per_class) * 0.1)

        if feature == 'classimb':
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = x_trn[subset_idxs]
                    y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
                else:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
                    y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))

        elif feature == 'noise':
            noise_size = int(len(y_trn) * 0.8)
            noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
            y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)

        min_samples = int(np.min(val_samples_per_class))
        for i in range(num_cls):
            if i == 0:
                subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                x_val_new = x_val[subset_ids]
                y_val_new = y_val[subset_ids].reshape(-1, 1)
            else:
                subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                x_val_new = np.row_stack((x_val_new, x_val[subset_ids]))
                y_val_new = np.row_stack((y_val_new, y_val[subset_ids].reshape(-1, 1)))
        min_samples = int(np.min(tst_samples_per_class))
        for i in range(num_cls):
            if i == 0:
                subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                x_tst_new = x_tst[subset_ids]
                y_tst_new = y_tst[subset_ids].reshape(-1, 1)
            else:
                subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                x_tst_new = np.row_stack((x_tst_new, x_tst[subset_ids]))
                y_tst_new = np.row_stack((y_tst_new, y_tst[subset_ids].reshape(-1, 1)))

        x_val = x_val_new
        y_val = y_val_new
        x_tst = x_tst_new
        y_tst = y_tst_new
        if feature == 'classimb':
            y_trn = y_trn_new
            x_trn = x_trn_new

        fullset = (x_trn, y_trn.reshape(-1))
        valset = (x_val, y_val.reshape(-1))
        testset = (x_tst, y_tst.reshape(-1))
        return fullset, valset, testset, num_cls

    elif dset_name == "usps":
        trn_file = os.path.join(datadir, 'usps.trn_full')
        tst_file = os.path.join(datadir, 'usps.tst')
        data_dims = 256
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        #fullset = (x_trn, y_trn)
        #valset = (x_val, y_val)
        #testset = (x_tst, y_tst)
        if feature == 'classimb':
            samples_per_class = np.zeros(num_cls)
            val_samples_per_class = np.zeros(num_cls)
            tst_samples_per_class = np.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(np.where(y_trn == i)[0])
                val_samples_per_class[i] = len(np.where(y_val == i)[0])
                tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
            min_samples = int(np.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = x_trn[subset_idxs]
                    y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
                else:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
                    y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))
            min_samples = int(np.min(val_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = x_val[subset_ids]
                    y_val_new = y_val[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = np.row_stack((x_val_new, x_val[subset_ids]))
                    y_val_new = np.row_stack((y_val_new, y_val[subset_ids].reshape(-1, 1)))
            min_samples = int(np.min(tst_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = x_tst[subset_ids]
                    y_tst_new = y_tst[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = np.row_stack((x_tst_new, x_tst[subset_ids]))
                    y_tst_new = np.row_stack((y_tst_new, y_tst[subset_ids].reshape(-1, 1)))
            x_val = x_val_new
            y_val = y_val_new
            x_tst = x_tst_new
            y_tst = y_tst_new
            y_trn = y_trn_new
            x_trn = x_trn_new
        elif feature == 'noise':
            noise_size = int(len(y_trn) * 0.8)
            noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
            y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)
        fullset = (x_trn, y_trn.reshape(-1))
        valset = (x_val, y_val.reshape(-1))
        testset = (x_tst, y_tst.reshape(-1))
        return fullset, valset, testset, num_cls

    elif dset_name == "ijcnn1":
        trn_file = os.path.join(datadir, 'ijcnn1.trn')
        val_file = os.path.join(datadir, 'ijcnn1.val')
        tst_file = os.path.join(datadir, 'ijcnn1.tst')
        data_dims = 22
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        # The class labels are (-1,1). Make them to (0,1)
        y_trn[y_trn < 0] = 0
        y_val[y_val < 0] = 0
        y_tst[y_tst < 0] = 0    

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        samples_per_class = np.zeros(num_cls)
        val_samples_per_class = np.zeros(num_cls)
        tst_samples_per_class = np.zeros(num_cls)
        for i in range(num_cls):
            samples_per_class[i] = len(np.where(y_trn == i)[0])
            val_samples_per_class[i] = len(np.where(y_val == i)[0])
            tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
        min_samples = int(np.min(samples_per_class) * 0.1)

        if feature == 'classimb':
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = x_trn[subset_idxs]
                    y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
                else:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
                    y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))

            min_samples = int(np.min(val_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    y_class = np.where(y_val == i)[0]
                    subset_ids = np.random.choice(y_class, size=min(3*min_samples,y_class.shape[0]), replace=False)
                    x_val_new = x_val[subset_ids]
                    y_val_new = y_val[subset_ids].reshape(-1, 1)
                else:
                    y_class = np.where(y_val == i)[0]
                    subset_ids = np.random.choice(y_class, size=min(3*min_samples,y_class.shape[0]), replace=False)
                    x_val_new = np.row_stack((x_val_new, x_val[subset_ids]))
                    y_val_new = np.row_stack((y_val_new, y_val[subset_ids].reshape(-1, 1)))
            min_samples = int(np.min(tst_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    y_class = np.where(y_tst == i)[0]
                    subset_ids = np.random.choice(y_class, size=min(3*min_samples,y_class.shape[0]), replace=False)
                    x_tst_new = x_tst[subset_ids]
                    y_tst_new = y_tst[subset_ids].reshape(-1, 1)
                else:
                    y_class = np.where(y_tst == i)[0]
                    subset_ids = np.random.choice(y_class, size=min(3*min_samples,y_class.shape[0]), replace=False)
                    x_tst_new = np.row_stack((x_tst_new, x_tst[subset_ids]))
                    y_tst_new = np.row_stack((y_tst_new, y_tst[subset_ids].reshape(-1, 1)))

            x_val = x_val_new
            y_val = y_val_new
            x_tst = x_tst_new
            y_tst = y_tst_new
            y_trn = y_trn_new
            x_trn = x_trn_new

        elif feature == 'noise':
            noise_size = int(len(y_trn) * 0.8)
            noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
            y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)


        fullset = (x_trn, y_trn.reshape(-1))
        valset = (x_val, y_val.reshape(-1))
        testset = (x_tst, y_tst.reshape(-1))
        return fullset, valset, testset, num_cls

    elif dset_name == "mnist":
        mnist_transform = transforms.Compose([            
            torchvision.transforms.ToTensor(),
            transforms.Grayscale(3),
            torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ])
        num_cls = 10
        fullset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
        return fullset, testset, num_cls

    elif dset_name == "fashion-mnist":
        mnist_transform = transforms.Compose([            
            torchvision.transforms.ToTensor(),
            transforms.Grayscale(3),
            torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ])
        num_cls = 10
        fullset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=mnist_transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=mnist_transform)
        return fullset, testset, num_cls

    elif dset_name == "sklearn-digits":
        np.random.seed(42)
        data, target = datasets.load_digits(return_X_y=True)
        # Test data is 10%
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        num_cls = 10
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        if feature == 'classimb':
            samples_per_class = np.zeros(num_cls)
            val_samples_per_class = np.zeros(num_cls)
            tst_samples_per_class = np.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(np.where(y_trn == i)[0])
                val_samples_per_class[i] = len(np.where(y_val == i)[0])
                tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
            min_samples = int(np.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = x_trn[subset_idxs]
                    y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
                else:
                    if i in selected_classes:
                        subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
                    else:
                        subset_idxs = np.where(y_trn == i)[0]
                    x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
                    y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))
            min_samples = int(np.min(val_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = x_val[subset_ids]
                    y_val_new = y_val[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_val == i)[0], size=min_samples, replace=False)
                    x_val_new = np.row_stack((x_val_new, x_val[subset_ids]))
                    y_val_new = np.row_stack((y_val_new, y_val[subset_ids].reshape(-1, 1)))
            min_samples = int(np.min(tst_samples_per_class))
            for i in range(num_cls):
                if i == 0:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = x_tst[subset_ids]
                    y_tst_new = y_tst[subset_ids].reshape(-1, 1)
                else:
                    subset_ids = np.random.choice(np.where(y_tst == i)[0], size=min_samples, replace=False)
                    x_tst_new = np.row_stack((x_tst_new, x_tst[subset_ids]))
                    y_tst_new = np.row_stack((y_tst_new, y_tst[subset_ids].reshape(-1, 1)))
            x_val = x_val_new
            y_val = y_val_new
            x_tst = x_tst_new
            y_tst = y_tst_new
            y_trn = y_trn_new
            x_trn = x_trn_new
        elif feature == 'noise':
            noise_size = int(len(y_trn) * 0.8)
            noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
            y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)
        fullset = (x_trn, y_trn.reshape(-1))
        valset = (x_val, y_val.reshape(-1))
        testset = (x_tst, y_tst.reshape(-1))
        return fullset, valset, testset, num_cls

    elif dset_name == "bc":
        data, target = datasets.load_breast_cancer(return_X_y=True)
        # Test data is 10%
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        num_cls = 2

        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)
        return fullset, valset, testset, num_cls

    elif dset_name == "blobs":
        data, target = make_blobs(n_samples=500, centers=2, n_features=2, random_state=42)
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)
        num_cls = 2
        # sc = StandardScaler()
        # x_trn = sc.fit_transform(x_trn)
        # x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls


def get_toydata(num_samples, num_features, num_classes, class_separation, plot_savepath):        

    data, target = make_classification(n_samples=num_samples, n_features=num_features, 
        n_informative=num_features, n_redundant=0, n_classes=num_classes, 
        n_clusters_per_class=1, class_sep=class_separation, random_state=42)
    
    x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, 
        test_size=0.1, random_state=42)

    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, 
        test_size=0.1, random_state=42)
    
    sc = StandardScaler()
    x_trn = sc.fit_transform(x_trn)
    x_val = sc.transform(x_val)
    x_tst = sc.transform(x_tst)

    trnset = (x_trn, y_trn)
    valset = (x_val, y_val)
    tstset = (x_tst, y_tst)

    # Plot only if the data is 2-D
    if num_features == 2:
        X_0 = x_trn[y_trn == 0]
        X_1 = x_trn[y_trn == 1]

        V_0 = x_val[y_val == 0]
        V_1 = x_val[y_val == 1]

        T_0 = x_tst[y_tst == 0]
        T_1 = x_tst[y_tst == 1]

        plt.figure()
        plt.scatter(X_0[:,0], X_0[:,1], color='blue', label='trn 0')
        plt.scatter(X_1[:,0], X_1[:,1], color='red', label='trn 1')

        plt.scatter(V_0[:,0], V_0[:,1], color='#CEF6F5', label='val 0')
        plt.scatter(V_1[:,0], V_1[:,1], color='#F5A9BC', label='val 1')

        plt.scatter(T_0[:,0], T_0[:,1], color='#8181F7', label='tst 0')
        plt.scatter(T_1[:,0], T_1[:,1], color='#8A0868', label='tst 1')
        
        plt.legend()
        plot_title = '2DtoyData' + '_' + str(num_samples) + '_' + str(num_classes) + '_' + str(class_separation) 
        plt.title(plot_title)
        plt.savefig(plot_savepath)
    
    return trnset, valset, tstset, num_classes
