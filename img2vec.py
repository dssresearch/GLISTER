import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
net.eval()
print(net)


torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)

layer = net._modules.get("avgpool")

net = net.to(device)

#if device == 'cuda':
#    net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
transform_train = transforms.Compose([
    transforms.Scale((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Scale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
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

train(1, trainloader, "cifar10.trn", True)
train(1, testloader, "cifar10.tst", False)
