'''
    Author : Ayush Dobhal
    Date created : 4/12/2020
    Description : This file contains code to convert an image dataset to vectorized format.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from models import *
from augmented_dataset import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Hardcoded to run on cuda
device = 'cuda'

pytorch_dataset = "CIFAR10"
dataset_name = "cifar10"
is_augmented_data = False
augmented_data_path = "/glusterfs/data/data_subset_selection/augmented_data/cifar10/cifar10_4X_full_dataset_augmented.pt"
curr_train_batch_size = 128
net = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
net.eval()

layer = net._modules.get("avgpool")

net = net.to(device)

#if device == 'cuda':
#    net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# CIFAR transform
cifar_transform = transforms.Compose([
    transforms.Scale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MNIST transform
mnist_transform = transforms.Compose([
    transforms.Grayscale(3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
     (0.1307,), (0.3081,))
])

if pytorch_dataset == "MNIST":
    if is_augmented_data == True:
        print("Using augmented data")
        augmented_data = torch.load(augmented_data_path)
        print(len(augmented_data))
        augmented_dataset = AugmentedTrainingDataset(augmented_data)
    else:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
elif pytorch_dataset == "CIFAR10":
    if is_augmented_data == True:
        print("Using augmented data")
        augmented_data = torch.load(augmented_data_path)
        print(len(augmented_data))
        augmented_dataset = AugmentedTrainingDataset(augmented_data)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
elif pytorch_dataset == "CIFAR100":
    if is_augmented_data == True:
        print("Using augmented data")
        augmented_data = torch.load(augmented_data_path)
        print(len(augmented_data))
        augmented_dataset = AugmentedTrainingDataset(augmented_data)
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar_transform)

if is_augmented_data == True:
    trainloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=curr_train_batch_size, shuffle=True, num_workers=2)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
                            
    def forward(self, x):
        x = self.features(x)
        return x

netm = ResNet50Bottom(net)

def train(epoch, dataloader, filename, isVal=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_size_trn = 128
    batch_size_tst = 100
    trn_file = open(filename, "a")
    val_file = open(filename+".val", "a")

    rows = len(dataloader)
    val_size = int(rows*0.1)
    
    indexes = np.random.randint(low=0, high=rows-1, size=int(val_size))
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        my_embedding = torch.zeros(inputs.shape[0], 512)
        def fun(m, i, o): my_embedding.copy_(o.data.squeeze())
        h = layer.register_forward_hook(fun)
        h_x = net(inputs)
        h.remove()
        np_embedding = my_embedding.cpu().detach().numpy()
        np_targets = targets.cpu().detach().numpy()
        for idx in range(inputs.shape[0]):
            val_idx = batch_idx * batch_size_trn + idx
            x_arrstr = np.char.mod('%f', np_embedding[idx])
            if isVal is True and val_idx in indexes:
                val_file.write(" ".join(x_arrstr)+" "+str(np_targets[idx])+"\n")
            else:
                trn_file.write(" ".join(x_arrstr)+" "+str(np_targets[idx])+"\n")

        #def copy_data(m, i, o):
        #    my_embedding.copy_(o.data)

        #h = layer.register_forward_hook(copy_data)
    trn_file.close()
    val_file.close()

train(1, trainloader, dataset_name+".trn", True)
#train(1, testloader, dataset_name+".tst")
