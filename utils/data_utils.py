import numpy as np
import os
import pandas as pd
import torch
import torchvision
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms


## Custom PyTorch Dataset Class wrapper
class CustomDataset(Dataset):
    def __init__(self, data, target, transform=None, device=None):       
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.to(device)
            self.targets = data.to(device)
        else:
            self.data = data
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
        return (sample_data, label)



## Utility function to load datasets from libsvm datasets
def libsvm_file_load(path,dim, save_data=False):
    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i for i in line.strip().split(",")]
            target.append(int(float(temp[-1])))  # Class Number. # Not assumed to be in (0, K-1)
            temp_data = [0] * dim
            count = 0
            for i in temp[:-1]:
                # ind, val = i.split(':')
                temp_data[count] = float(i)
                count += 1
            data.append(temp_data)
            line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)


"""
    data = []
    target = []
    with open(path) as fp:
       line = fp.readline()
       while line:
        temp = [i for i in line.strip().split(" ")]
        target.append(float(temp[0])) # Class Number. # Not assumed to be in (0, K-1)
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
"""

## Utility function to load datasets from libsvm datasets
## Used in: KnnSubmod Selection
## DEPRECATED
def libsvm_to_kNNstandard(datadir, filename, dim, split_val=False):
    data = []
    target = []
    filepath = os.path.join(datadir, filename)    
    with open(filepath) as fp:
       line = fp.readline()
       while line:
        temp = [i for i in line.strip().split(" ")]
        row_vector = [0] * (dim+1) # +1 for the y label
        row_label = int(temp[0])
        # Label not assumed to be in (0, K-1) since that is not necessary for kNNsubmod
        target.append(row_label) # Class Number. 
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
    if split_val:
        train, val = train_test_split(all_data, test_size=0.1, random_state=42)
        train_path = os.path.join(datadir, 'knndata_' + filename + '.trn')
        val_path = os.path.join(datadir, 'knndata_' + filename + '.val')        
        np.savetxt(train_path, train, fmt='%.6f')
        np.savetxt(val_path, val, fmt='%.6f')
    else:
        data_path = os.path.join(datadir, 'knndata_' + filename)
        np.savetxt(data_path, all_data, fmt='%.6f')   # entire data
    # return all_data


def numpyarr_to_kNNstandard(x_data, y_data):
    # basically thr y_data (labels) should be the last column!
    data = np.c_[x_data, y_data]
    return data


## Function to load a discrete UCI dataset and make it ordinal.
def clean_uci_ordinal_data(inp_fname, out_fname):
    # trn, val, tst split: 0.8, 0.1, 0.1
    data = np.genfromtxt(inp_fname, delimiter=',', dtype='str')
    enc = OrdinalEncoder(dtype=int)
    enc.fit(data)
    transformed_data = enc.transform(data)
    # np.random.shuffle(transformed_data)     # randomly shuffle the data points
    
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
def write_knndata(datadir, dset_name):
    fullset, valset, tstset, num_cls = load_dataset_numpy(datadir, dset_name)
    x_trn, y_trn = fullset
    x_val, y_val = valset
    x_tst, y_tst = tstset
    ## Create VAL data
    #x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

    trndata = np.c_[x_trn, y_trn]
    valdata = np.c_[x_val, y_val]
    tstdata = np.c_[x_tst, y_tst]

    # Write out the trndata
    trn_filepath = os.path.join(datadir, 'knn_' + dset_name + '.trn')
    val_filepath = os.path.join(datadir, 'knn_' + dset_name + '.val')
    tst_filepath = os.path.join(datadir, 'knn_' + dset_name + '.tst')

    np.savetxt(trn_filepath, trndata, fmt='%.6f')
    np.savetxt(val_filepath, valdata, fmt='%.6f')
    np.savetxt(tst_filepath, tstdata, fmt='%.6f')

    return



## If splitting the data -- make a deterministic change!
## Takes in a dataset name and returns a Tuple of ((x_trn, y_trn), (x_tst, y_tst), num_cls)
def load_dataset_numpy(datadir, dset_name):
    if dset_name == "dna":
        trn_file = os.path.join(datadir, 'dna.scale.trn')
        val_file = os.path.join(datadir, 'dna.scale.val')
        tst_file = os.path.join(datadir, 'dna.scale.tst')
        data_dims = 180
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        x_trn = np.concatenate((x_trn, x_val))
        y_trn = np.concatenate((y_trn, y_val))
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls

    elif dset_name == "linsep":
        trn_file = os.path.join(datadir, 'linsep.trn')
        val_file = os.path.join(datadir, 'linsep.val')
        tst_file = os.path.join(datadir, 'linsep.tst')
        data_dims = 2
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        # x_trn = np.concatenate((x_trn, x_val))
        # y_trn = np.concatenate((y_trn, y_val))
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)
        return fullset, valset, testset, num_cls

    elif dset_name == "linsep_4":
        trn_file = os.path.join(datadir, 'linsep_4.trn')
        val_file = os.path.join(datadir, 'linsep_4.val')
        tst_file = os.path.join(datadir, 'linsep_4.tst')
        data_dims = 2
        num_cls = 4
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        # x_trn = np.concatenate((x_trn, x_val))
        # y_trn = np.concatenate((y_trn, y_val))
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)
        return fullset, valset, testset, num_cls

    elif dset_name == "gauss_2":
        trn_file = os.path.join(datadir, 'gauss_2.trn')
        val_file = os.path.join(datadir, 'gauss_2.val')
        tst_file = os.path.join(datadir, 'gauss_2.tst')
        data_dims = 2
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        # x_trn = np.concatenate((x_trn, x_val))
        # y_trn = np.concatenate((y_trn, y_val))
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)
        return fullset, valset, testset, num_cls

    elif dset_name == "clf_2":
        trn_file = os.path.join(datadir, 'clf_2.trn')
        val_file = os.path.join(datadir, 'clf_2.val')
        tst_file = os.path.join(datadir, 'clf_2.tst')
        data_dims = 2
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        # x_trn = np.concatenate((x_trn, x_val))
        # y_trn = np.concatenate((y_trn, y_val))
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        valset = (x_val, y_val)
        testset = (x_tst, y_tst)
        return fullset, valset, testset, num_cls

    elif dset_name == "letter":
        data_dims = 16
        num_cls = 26
        csvfile = os.path.join(datadir, 'letters.txt')
        letters = pd.read_csv(csvfile)
        x_trn = np.array(letters[:15000].drop(['letter'], 1))
        x_tst = np.array(letters[15000:].drop(['letter'], 1))

        training_labels = np.array(letters[:15000]['letter'])
        test_labels = np.array(letters[15000:]['letter']) 

        enc = OrdinalEncoder(dtype=int)
        enc.fit(training_labels.reshape((-1,1)))
        y_trn = enc.transform(training_labels.reshape(-1,1))
        y_tst = enc.transform(test_labels.reshape(-1,1))
        y_trn = y_trn.reshape(-1)
        y_tst = y_tst.reshape(-1)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "pendigits":
        trn_file = os.path.join(datadir, 'pendigits.trn_full')
        tst_file = os.path.join(datadir, 'pendigits.tst')
        data_dims = 16
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)

        return fullset, testset, num_cls
    elif dset_name == "satimage":
        trn_file = os.path.join(datadir, 'satimage.scale.trn')
        val_file = os.path.join(datadir, 'satimage.scale.val')
        tst_file = os.path.join(datadir, 'satimage.scale.tst')
        data_dims = 36
        num_cls = 6
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        x_trn = np.concatenate((x_trn, x_val))
        y_trn = np.concatenate((y_trn, y_val))
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "svmguide1":
        trn_file = os.path.join(datadir, 'svmguide1.trn_full')
        tst_file = os.path.join(datadir, 'svmguide1.tst')
        data_dims = 4
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "usps":
        trn_file = os.path.join(datadir, 'usps.trn_full')
        tst_file = os.path.join(datadir, 'usps.tst')
        data_dims = 256
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "ijcnn1":
        trn_file = os.path.join(datadir, 'ijcnn1.trn')
        val_file = os.path.join(datadir, 'ijcnn1.val')
        tst_file = os.path.join(datadir, 'ijcnn1.tst')
        data_dims = 22
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        x_trn = np.concatenate((x_trn, x_val))
        y_trn = np.concatenate((y_trn, y_val))
        # The class labels are (-1,1). Make them to (0,1)
        y_trn[y_trn < 0] = 0
        y_tst[y_tst < 0] = 0        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "mnist":
        mnist_transform = transforms.Compose([            
            torchvision.transforms.ToTensor(),
            transforms.Grayscale(3),
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
        num_cls = 10
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "bc":
        data, target = datasets.load_breast_cancer(return_X_y=True)
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)
        num_cls = 2
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        fullset = (x_trn, y_trn)
        testset = (x_tst, y_tst)
        return fullset, testset, num_cls



## Takes in a dataset name and returns a PyTorch Dataset Object
def load_dataset_pytorch(datadir, dset_name, feature=None):
    if dset_name == "dna":
        trn_file = os.path.join(datadir, 'dna.scale.trn')
        val_file = os.path.join(datadir, 'dna.scale.val')
        tst_file = os.path.join(datadir, 'dna.scale.tst')
        data_dims = 180
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        x_trn = np.concatenate((x_trn, x_val))
        y_trn = np.concatenate((y_trn, y_val))
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        # Convert to pytorch tensors
        x_trn = torch.from_numpy(x_trn.astype(np.float32))
        x_tst = torch.from_numpy(x_tst.astype(np.float32))
        y_trn = torch.from_numpy(y_trn.astype(np.int64))
        y_tst = torch.from_numpy(y_tst.astype(np.int64))
        fullset = CustomDataset(x_trn, y_trn)
        testset = CustomDataset(x_tst, y_tst)
        return fullset, testset, num_cls 
    elif dset_name == "letter":
        trn_file = os.path.join(datadir, 'letter.scale.trn')
        val_file = os.path.join(datadir, 'letter.scale.val')
        tst_file = os.path.join(datadir, 'letter.scale.tst')
        data_dims = 16
        num_cls = 26 
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        x_trn = np.concatenate((x_trn, x_val))
        y_trn = np.concatenate((y_trn, y_val))
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        # Convert to pytorch tensors
        x_trn = torch.from_numpy(x_trn.astype(np.float32))
        x_tst = torch.from_numpy(x_tst.astype(np.float32))
        y_trn = torch.from_numpy(y_trn.astype(np.int64))
        y_tst = torch.from_numpy(y_tst.astype(np.int64))
        fullset = CustomDataset(x_trn, y_trn)
        testset = CustomDataset(x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "pendigits":
        trn_file = os.path.join(datadir, 'pendigits.trn_full')
        tst_file = os.path.join(datadir, 'pendigits.tst')
        data_dims = 16
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        # Convert to pytorch tensors
        x_trn = torch.from_numpy(x_trn.astype(np.float32))
        x_tst = torch.from_numpy(x_tst.astype(np.float32))
        y_trn = torch.from_numpy(y_trn.astype(np.int64))
        y_tst = torch.from_numpy(y_tst.astype(np.int64))
        fullset = CustomDataset(x_trn, y_trn)
        testset = CustomDataset(x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "satimage":
        trn_file = os.path.join(datadir, 'satimage.scale.trn')
        val_file = os.path.join(datadir, 'satimage.scale.val')
        tst_file = os.path.join(datadir, 'satimage.scale.tst')
        data_dims = 36
        num_cls = 6
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        x_trn = np.concatenate((x_trn, x_val))
        y_trn = np.concatenate((y_trn, y_val))
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        # Convert to pytorch tensors
        x_trn = torch.from_numpy(x_trn.astype(np.float32))
        x_tst = torch.from_numpy(x_tst.astype(np.float32))
        y_trn = torch.from_numpy(y_trn.astype(np.int64))
        y_tst = torch.from_numpy(y_tst.astype(np.int64))
        fullset = CustomDataset(x_trn, y_trn)
        testset = CustomDataset(x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "svmguide1":
        trn_file = os.path.join(datadir, 'svmguide1.trn_full')
        tst_file = os.path.join(datadir, 'svmguide1.tst')
        data_dims = 4
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        # Convert to pytorch tensors
        x_trn = torch.from_numpy(x_trn.astype(np.float32))
        x_tst = torch.from_numpy(x_tst.astype(np.float32))
        y_trn = torch.from_numpy(y_trn.astype(np.int64))
        y_tst = torch.from_numpy(y_tst.astype(np.int64))
        fullset = CustomDataset(x_trn, y_trn)
        testset = CustomDataset(x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "usps":
        trn_file = os.path.join(datadir, 'usps.trn_full')
        tst_file = os.path.join(datadir, 'usps.tst')
        data_dims = 256
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        # Convert to pytorch tensors
        x_trn = torch.from_numpy(x_trn.astype(np.float32))
        x_tst = torch.from_numpy(x_tst.astype(np.float32))
        y_trn = torch.from_numpy(y_trn.astype(np.int64))
        y_tst = torch.from_numpy(y_tst.astype(np.int64))
        fullset = CustomDataset(x_trn, y_trn)
        testset = CustomDataset(x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "ijcnn1":
        trn_file = os.path.join(datadir, 'ijcnn1.trn')
        val_file = os.path.join(datadir, 'ijcnn1.val')
        tst_file = os.path.join(datadir, 'ijcnn1.tst')
        data_dims = 22
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        x_trn = np.concatenate((x_trn, x_val))
        y_trn = np.concatenate((y_trn, y_val))
        # The class labels are (-1,1). Make them to (0,1)
        y_trn[y_trn < 0] = 0
        y_tst[y_tst < 0] = 0
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        # Convert to pytorch tensors
        x_trn = torch.from_numpy(x_trn.astype(np.float32))
        x_tst = torch.from_numpy(x_tst.astype(np.float32))
        y_trn = torch.from_numpy(y_trn.astype(np.int64))
        y_tst = torch.from_numpy(y_tst.astype(np.int64))
        fullset = CustomDataset(x_trn, y_trn)
        testset = CustomDataset(x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "mnist":
        mnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist_val_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        num_cls = 10
        fullset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
        if feature=='classimb':
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples, replace=False))
                    else:
                        subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples, replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)
        valset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_val_transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
        return fullset, valset, testset, num_cls
    elif dset_name == "cifar10":
        cifar_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        cifar_val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        num_cls = 10
        fullset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=cifar_transform)
        valset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=cifar_val_transform)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=cifar_transform)
        if feature == 'classimb':
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)
        return fullset, valset, testset, num_cls
    elif dset_name == "sklearn-digits":
        data, target = datasets.load_digits(return_X_y=True)
        # Test data is 10%
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        num_cls = 10
        # Convert to pytorch tensors
        x_trn = torch.from_numpy(x_trn.astype(np.float32))
        x_tst = torch.from_numpy(x_tst.astype(np.float32))
        y_trn = torch.from_numpy(y_trn.astype(np.int64))
        y_tst = torch.from_numpy(y_tst.astype(np.int64))
        fullset = CustomDataset(x_trn, y_trn)
        testset = CustomDataset(x_tst, y_tst)
        return fullset, testset, num_cls
    elif dset_name == "bc":
        data, target = datasets.load_breast_cancer(return_X_y=True)
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_tst = sc.transform(x_tst)
        num_cls = 2
        # Convert to pytorch tensors
        x_trn = torch.from_numpy(x_trn.astype(np.float32))
        x_tst = torch.from_numpy(x_tst.astype(np.float32))
        y_trn = torch.from_numpy(y_trn.astype(np.int64))
        y_tst = torch.from_numpy(y_tst.astype(np.int64))
        fullset = CustomDataset(x_trn, y_trn)
        testset = CustomDataset(x_tst, y_tst)
        return fullset, testset, num_cls




## Returns TRN, VAL, TST Numpy Arrays with TRN data (y-labels) being corrupted
## noise_fraction part of TRN labels are chosen and their labels are randomly
## flipped to any of the K classes for the given dataset. 
## Assumes noise_fractions is between 0 and 1
def get_noisy_data_numpy(datadir, dset_name, noise_fraction):
    if noise_fraction < 0 or noise_fraction > 1:
        print("Incorrect Value for Noise Fraction. Should be in range (0,1)")
        return

    (x_trn, y_trn), (x_tst, y_tst), num_cls = load_dataset_numpy(datadir, dset_name)
    # Separate out the Validation Data using the same fraction = 0.1 and random_state
    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

    n_trn = y_trn.shape[0]
    n_flip = int(noise_fraction * n_trn)
    # choose indices to flip labels.
    flip_idxs = np.random.choice(range(n_trn), size=n_flip,replace=False)

    y_trn_org = np.copy(y_trn)
    y_trn[flip_idxs] = np.random.choice(range(num_cls), size=n_flip, replace=True)

    wrong_indices = np.where(y_trn != y_trn_org)[0]
    check = sum(wrong_indices == flip_idxs)
    assert(check == n_flip)
    wrong_labels = y_trn[wrong_indices]

    noisy_trndata = (x_trn, y_trn)
    valdata = (x_val, y_val)
    tstdata = (x_tst, y_tst)

    ## Not storing the noisy data!
    ## Also return the original clean and uncorrupted labels
    return noisy_trndata, valdata, tstdata, y_trn_org
    




#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--input", required=True, help="input file")
#ap.add_argument("-o", "--output", required=True, help="output file")
#args = vars(ap.parse_args())
#
#clean_data(args['input'], args['output'])
#



