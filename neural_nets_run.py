'''
    Author : Ayush Dobhal
    Date created : 02/25/2020
    Description : This file contains code for running neural nets execution based on the *.input file.
    Currently supported datasets : CIFAR10, CIFAR100 and MNIST
'''
import argparse
import datetime
import json
import math
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from augmented_dataset import *
from models import *
from torch.utils.data.sampler import SubsetRandomSampler

''' TODO
- clean this up a bit like standard deep learning experiment repo
- write out the utility functions in a separate file
- have a single main function which does the experiment run given the config file.
'''
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filepath", required=True, type=str, help="input file")
arguments = parser.parse_args()
input_file = arguments.filepath
print("Using input file : "+input_file+"\n")
#input_file = "./neural_net_run.input"   # Config File for the experiment
with open(input_file) as f:
    input_dict = json.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.cuda.empty_cache()
device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
# Cifar dataset transforms
cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Mnist transform: Same for train and test since we dont do any random data augmentation
mnist_transform = transforms.Compose([
    transforms.Grayscale(3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
     (0.1307,), (0.3081,))
])


budgets = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75]    # Training set fractions to Select
indices_file = input_dict["indices_file"]
checkpoint_file = input_dict["checkpoint"]
checkpoint_save_name = input_dict["checkpoint_save_name"]
learning_rate = input_dict["learning_rate"]
momentum_rate = input_dict["momentum_rate"]
num_epoch = input_dict["num_epoch"]
resume = input_dict["resume"]
full_dataset = input_dict["full_dataset"]
random_subset = input_dict["random_subset"]
current_budget = input_dict["current_budget"]
pytorch_dataset = input_dict["pytorch_dataset"]
num_classes = input_dict["num_classes"]
is_augmented_data = input_dict["is_augmented_data"]
augmented_data_path = input_dict["augmented_data_path"]
use_subset = input_dict["use_subset"]

logfile = open(input_dict["logfile_path"], 'a')
logfile.write("logfile = "+input_dict["logfile_path"]+"\n")
curr_train_batch_size = 128
curr_test_batch_size = 100

logfile.write("Using "+pytorch_dataset+" dataset for cnn\n")
logfile.write("Using initial learning rate "+str(learning_rate)+"\n")
if pytorch_dataset == "cifar10":
    if is_augmented_data == 1:
        print("Using augmented data")
        augmented_data = torch.load(augmented_data_path)
        print(len(augmented_data))
        augmented_dataset = AugmentedTrainingDataset(augmented_data)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform_test)
elif pytorch_dataset == "cifar100":
    print("Using CIFAR100\n")
    if is_augmented_data == 1:
        print("Using augmented data")
        augmented_data = torch.load(augmented_data_path)
        print(len(augmented_data))
        augmented_dataset = AugmentedTrainingDataset(augmented_data)
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar_transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar_transform_test)
elif pytorch_dataset == "mnist":
    print("Using MNIST\n")
    if is_augmented_data == 1:
        print("Using augmented data")
        augmented_data = torch.load(augmented_data_path)
        print(len(augmented_data))
        augmented_dataset = AugmentedTrainingDataset(augmented_data)
    else:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

if is_augmented_data != 1:
    num_samples_train = trainset.data.shape[0]  # number of examples in training set

if full_dataset != 1:   # using a subset of given budget
    if random_subset == 1:  # random subset
        logfile.write("Using random subset indices with budget = "+str(current_budget)+"\n")
        subset_indices = np.random.randint(low=0, high=num_samples_train-1, size=int(num_samples_train*current_budget))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=curr_train_batch_size, shuffle=False, 
                             sampler=SubsetRandomSampler(subset_indices), num_workers=2)
    elif is_augmented_data == 1:
        subset_indices = np.genfromtxt(indices_file, delimiter=',', dtype=int)
        logfile.write("Submod indices_file = "+indices_file+"\n")
        print("Augmented data with subset : ", len(subset_indices))
        trainloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=curr_train_batch_size, shuffle=False,
                             sampler=SubsetRandomSampler(subset_indices), num_workers=2)
    else:   # selected subset
        subset_indices = np.genfromtxt(indices_file, delimiter=',', dtype=int)
        logfile.write("Submod indices_file = "+indices_file+"\n")
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=curr_train_batch_size, shuffle=False, 
                             sampler=SubsetRandomSampler(subset_indices), num_workers=2)

else:   # using the full dataset
     if is_augmented_data == 1:
        trainloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=curr_train_batch_size, shuffle=True, num_workers=2)
        num_samples_train = len(augmented_data)
     else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=curr_train_batch_size, shuffle=True, num_workers=2)
        num_samples_train = len(trainset)
     logfile.write("Using full dataset\n")
     print("Using full dataset\n")

print("Trainloader size : "+str(len(trainloader)))
testloader = torch.utils.data.DataLoader(testset, batch_size=curr_test_batch_size, shuffle=False, num_workers=2)

print('==> Building model..')
net = ResNet18(num_classes)     # Model

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if resume == 1:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    cpath = "./checkpoint/"+checkpoint_file
    if os.path.isdir('checkpoint') and os.path.exists(cpath):
        checkpoint = torch.load(cpath)
        net.load_state_dict(checkpoint['net'])
        best_acc = 0
        start_epoch = checkpoint['epoch']
        print("Resume from acc="+str(best_acc)+" epoch="+str(start_epoch)+"\n")
        logfile.write("Resume from acc="+str(best_acc)+" epoch="+str(start_epoch)+"\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum_rate, weight_decay=5e-4)
MILESTONES = [60, 120, 160]     # epochs to checkpoint at
warmup = 2
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup)

def adjust_learning_rate(optimizer, epoch, max_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    TRAINING_LR_INIT_EPOCHS = 10
    TRAINING_LR_MAX          = 0.001
    TRAINING_LR_INIT_SCALE   = 0.01
    TRAINING_LR_FINAL_SCALE  = 0.01
    TRAINING_LR_FINAL_EPOCHS = max_epoch - TRAINING_LR_INIT_EPOCHS
    TRAINING_NUM_EPOCHS = max_epoch
    TRAINING_LR_INIT    = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
    TRAINING_LR_FINAL   = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE
    lr = 0.1
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL

    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training Procedure
def train(epoch):
    print('\nEpoch: %d' % epoch)
    ts = datetime.datetime.now()
    print("\nEpoch train :"+str(epoch)+" started at "+ str(ts)+"\n")
    logfile.write("Epoch train : "+str(epoch)+" started at "+ str(ts)+"\n")
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    #adjust_learning_rate(optimizer, epoch, num_epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if (inputs.shape[0] != curr_train_batch_size):
            continue

        if epoch <= warmup:
            warmup_scheduler.step()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx != 0 and batch_idx % 100 == 0:
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, " Train : ",len(trainloader), ' Loss: %.3f | Acc: %.3f%% (%d/%d)' %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            logfile.write(' Batch :'+str(batch_idx) + " Trainloader len :"+ str(len(trainloader)) +' Loss: '+str((train_loss/(batch_idx+1)))+' | Acc: '+str(100.*correct/total)+' ('+str(correct)+'/'+str(total)+')'+'\n')

    if (epoch+1) % 50 == 0:
        print('Saving..')
        state = {
         'net': net.state_dict(),
         'epoch': epoch
        }
        ckp_dir = checkpoint_save_name 
        ckp_name = checkpoint_save_name+'_'+str(epoch+1)
        #if not os.path.isdir('checkpoint'):
        os.makedirs('checkpoint/'+ckp_dir, exist_ok=True)
        torch.save(state, './checkpoint/'+ckp_dir+'/'+ckp_name+'.pth')
    
    tf = datetime.datetime.now()
    print("Epoch train:"+str(epoch)+" finished at : "+str(tf)+"\n")
    logfile.write("Epoch train:"+str(epoch)+" finished at : "+str(tf)+"\n")
    logfile.write("Epoch train:"+str(epoch)+" time elapsed : "+str(tf-ts)+"\n")

def test(epoch):
    global best_acc
    ts = datetime.datetime.now()
    print("Epoch test :"+str(epoch)+" started at "+ str(ts)+"\n")
    logfile.write("Epoch test : "+str(epoch)+" started at "+ str(ts)+"\n")
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if (inputs.shape[0] != curr_test_batch_size):
                continue
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, " Test : ",len(testloader), ' Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            logfile.write('Batch :'+str(batch_idx) + " Testloader len :"+ str(len(trainloader)) +' Loss: '+str((test_loss/(batch_idx+1)))+' | Acc: '+str(100.*correct/total)+' ('+str(correct)+'/'+str(total)+')'+'\n')

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    #if acc > best_acc:
    #    print('Saving..')
    #    state = {
    #        'net': net.state_dict(),
    #        'acc': acc,
    #        'epoch': epoch,
    #    }
    #    if not os.path.isdir('checkpoint'):
    #        os.mkdir('checkpoint')
    #    torch.save(state, './checkpoint/'+checkpoint_file)
    #    best_acc = acc

    tf = datetime.datetime.now()
    print("Epoch test:"+str(epoch)+" finished at : "+str(tf)+"\n")
    logfile.write("Epoch test:"+str(epoch)+" finished at : "+str(tf)+"\n")
    logfile.write("Epoch test:"+str(epoch)+" time elapsed : "+str(tf-ts)+"\n")
    # function over

ts = datetime.datetime.now()
print("CNN run started at "+ str(ts)+"\n")
logfile.write("CNN run started at "+ str(ts)+"\n")
for epoch in range(start_epoch, num_epoch):
    if epoch > warmup:
        train_scheduler.step(epoch)
    train(epoch)

test(epoch)
tf = datetime.datetime.now()
print("CNN ended started at "+ str(tf)+"\n")
logfile.write("CNN run ended at "+ str(tf)+"\n")
logfile.write("Total time elapsed : "+ str(tf-ts)+"\n")
print("Total time elapsed : "+ str(tf-ts)+"\n")

logfile.write("Test accuracy : "+str(best_acc))

