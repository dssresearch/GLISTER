import numpy as np
#from dataset import get_dataset, get_handler
import torch.nn.functional as F
from torch import nn
#from torchvision import transforms
import torch
from badge.query_strategies.badge_sampling import BadgeSampling
#from sklearn.model_selection import train_test_split
#import resnet
#from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler
#from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
#from torchvision import datasets
from PIL import Image

import sys
sys.path.append('../')
import math
import random

#from utils.custom_dataset import CustomDataset_WithId, load_dataset_numpy, write_knndata
#from custom_dataset_old import load_dataset_numpy as load_dataset_numpy_old, write_knndata as write_knndata_old
#from utils.data_utils import load_dataset_pytorch
#from models.simpleNN_net import *
#from models.logistic_regression import LogisticRegNet

from utils.custom_dataset import load_dataset_custom, CustomDataset_WithId

# linear model class
class linMod(nn.Module):
    def __init__(self, dim,nClasses):
        super(linMod, self).__init__()
        self.dim = dim
        self.lm = nn.Linear(int(np.prod(dim)), nClasses)
    def forward(self, x):
        x = x.view(-1, int(np.prod(self.dim)))
        out = self.lm(x)
        return out, x
    def get_embedding_dim(self):
        return int(np.prod(self.dim))

# mlp model class
class mlpMod(nn.Module):
    def __init__(self, dim, nClasses,embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, nClasses)
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb
    def get_embedding_dim(self):
        return self.embSize

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            #x.dtype = np.uint8
            #x = Image.fromarray(x,mode = 'RGB')
            print(x.shape)
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)



def return_accuracies(start_idxs,NUM_ROUND,NUM_QUERY,epoch,learning_rate,datadir,data_name, feature):

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    fullset, valset, testset, data_dims, num_cls = load_dataset_custom(datadir, data_name,feature=feature, isnumpy=True)
    
    x_trn, y_trn = fullset
    x_val, y_val = valset
    x_tst, y_tst = testset

    x_trn, y_trn = torch.from_numpy(fullset[0]).float(), torch.from_numpy(fullset[1]).long()

    handler = CustomDataset_WithId
    
    args = {'transform':None,
            'n_epoch':epoch,
            'loader_tr_args':{'batch_size': NUM_QUERY},
            'loader_te_args':{'batch_size': 1000},
            'optimizer_args':{'lr': learning_rate},
            'transformTest':None}

    args['lr'] = learning_rate

    
    n_pool = len(y_trn)
    n_val = len(y_val)
    n_test = len(y_tst)

    net = mlpMod(x_trn.shape[1], num_cls,100) #linMod(x_trn.shape[1], num_cls)

    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_lb[start_idxs] = True

    strategy = BadgeSampling(x_trn, y_trn, idxs_lb, net, handler, args)

    strategy.train()

    unlabled_acc= np.zeros(NUM_ROUND+1)
    tst_acc = np.zeros(NUM_ROUND+1)
    val_acc = np.zeros(NUM_ROUND+1)

    P = strategy.predict(x_tst, y_tst)
    tst_acc[0] = 100.0 * P.eq(torch.tensor(y_tst)).sum().item()/ n_test
    print('\ttesting accuracy {}'.format(tst_acc[0]), flush=True)
    #tst_acc[0] = 100.0 * P.eq(torch.tensor(y_tst)).sum().item()/ n_test
    #print('\ttesting accuracy {}'.format(tst_acc[0]), flush=True)

    P = strategy.predict(x_val, y_val)
    val_acc[0] = 100.0 * P.eq(torch.tensor(y_val)).sum().item() / n_val    

    #idxs_unlabeled = (idxs_lb == False).nonzero().flatten().tolist()
    u_x_trn = x_trn[~idxs_lb]
    u_y_trn = y_trn[~idxs_lb]
    P = strategy.predict(u_x_trn, u_y_trn)
    unlabled_acc[0] = 100.0 * P.eq(torch.tensor(u_y_trn)).sum().item() / len(u_y_trn)   

    for rd in range(1, NUM_ROUND+1):
        print('Round {}'.format(rd), flush=True)

        # query
        output = strategy.query(NUM_QUERY)
        q_idxs = output
        idxs_lb[q_idxs] = True

        # report weighted accuracy
        #corr = (strategy.predict(X_tr[q_idxs], torch.Tensor(Y_tr.numpy()[q_idxs]).long())).numpy() == Y_tr.numpy()[q_idxs]

        # update
        strategy.update(idxs_lb)
        strategy.train()

        # round accuracy
        P = strategy.predict(x_tst, y_tst)
        tst_acc[rd] = 100.0 * P.eq(torch.tensor(y_tst)).sum().item() / n_test
        print(rd,'\ttesting accuracy {}'.format(tst_acc[rd]), flush=True)

        P = strategy.predict(x_val, y_val)
        val_acc[rd] = 100.0 * P.eq(torch.tensor(y_val)).sum().item() / n_val 

        #idxs_unlabeled = (idxs_lb == False).nonzero().flatten().tolist()
        u_x_trn = x_trn[~idxs_lb]
        u_y_trn = y_trn[~idxs_lb]
        P = strategy.predict(u_x_trn, u_y_trn)
        unlabled_acc[rd] = 100.0 * P.eq(torch.tensor(u_y_trn)).sum().item() / len(u_y_trn)
        
        #print(str(sum(idxs_lb)) + '\t' + 'unlabled data', len(u_y_trn),flush=True)
        
        if sum(~strategy.idxs_lb) < NUM_QUERY: 
            sys.exit('too few remaining points to query')

    return val_acc, tst_acc, unlabled_acc, np.arange(n_pool)[idxs_lb]
