'''
    Author : Ayush Dobhal
    Date created : 04/20/2020
    Description : This file contains code to create a dataset containing augmented dataset + original dataset by applying transforms. 
'''
import datetime
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from models import *
from torch.utils.data.sampler import SubsetRandomSampler

#from utils import progress_bar
input_file = "./neural_net_run.input"
with open(input_file) as f:
    input_dict = json.load(f)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.cuda.empty_cache()
device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train_default = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train1 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train2 = transforms.Compose([
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train3 = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train4 = transforms.Compose([
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train5 = transforms.Compose([
    transforms.RandomAffine((-45, 45)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train6 = transforms.Compose([
    transforms.RandomRotation((-60, 60)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

budgets = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75]
indices_file = input_dict["indices_file"]
checkpoint_file = input_dict["checkpoint"]
checkpoint_save_name = input_dict["checkpoint_save_name"]
learning_rate = input_dict["learning_rate"]
num_epoch = input_dict["num_epoch"]
resume = input_dict["resume"]
full_dataset = input_dict["full_dataset"]
random_subset = input_dict["random_subset"]
current_budget = input_dict["current_budget"]
pytorch_dataset = input_dict["pytorch_dataset"]
num_classes = input_dict["num_classes"]
num_samples = 50000

full_dataset = 1
pytorch_dataset = "cifar10"
#logfile = open(input_dict["logfile_path"], 'a')

#logfile.write("logfile = "+input_dict["logfile_path"]+"\n")
curr_train_batch_size = 128
curr_test_batch_size = 100

#logfile.write("Using "+pytorch_dataset+" dataset for cnn\n")

#subset_dict = {}
dataset_size = 0
if pytorch_dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
elif pytorch_dataset == "cifar100":
    print("Using CIFAR100\n")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)

if full_dataset != 1:
    subset_indices = np.genfromtxt(indices_file, delimiter=',', dtype=int)
    #for idx in range(len(subset_indices)):
    #    subset_dict[idx] = 1
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=curr_train_batch_size, shuffle=False, sampler=SubsetRandomSampler(subset_indices), num_workers=2)
    dataset_size = len(subset_indices)
else:
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=curr_train_batch_size, shuffle=True, num_workers=2)
    #logfile.write("Using full dataset\n")

print("Trainloader size : "+str(len(trainloader)))

ts = datetime.datetime.now()
#logfile.write("Epoch test : "+str(epoch)+" started at "+ str(ts)+"\n")
print("trainset stats")
print(len(trainset))
print(type(trainset))
print(len(trainloader))
#item0 = trainset.__getitem__(0)
#print("one sample from dataset")
#print(torchvision.transforms.ToTensor()(item0[0]))
#print(torchvision.transforms.ToTensor()(item0[0]).size())
#print(item0[0])
#print(type(item0))

print("transformed sample")
augmented_dataset = []

for i in range(dataset_size):
    if full_dataset != 1:
        idx = subset_indices[i]
    else:
        idx = i
    item0 = trainset.__getitem__(idx)
    label = item0[1]
    tdata = transform_train_default(item0[0])
    augmented_dataset.append((tdata, label))
    
    tdata = transform_train1(item0[0])
    augmented_dataset.append((tdata, label))

    tdata = transform_train2(item0[0])
    augmented_dataset.append((tdata, label))
    
    tdata = transform_train3(item0[0])
    augmented_dataset.append((tdata, label))
    
    #tdata = transform_train4(item0[0])
    #augmented_dataset.append((tdata, label))
    #
    #tdata = transform_train5(item0[0])
    #augmented_dataset.append((tdata, label))

    #tdata = transform_train6(item0[0])
    #augmented_dataset.append((tdata, label))

print("New augmented dataset created")
print(len(augmented_dataset))
torch.save(augmented_dataset, "cifar10_4N_augmented.pt")
#print(len(tdata))
#for batch_idx, (inputs, targets) in enumerate(testloader):
#    inputs, targets = inputs.to(device), targets.to(device)
#    if (inputs.shape[0] != curr_test_batch_size):
#        continue
