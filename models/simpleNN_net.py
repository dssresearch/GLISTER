import torch.nn as nn
import torch.nn.functional as F


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_units):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, num_classes)
    
    def forward(self, x):
        l1scores = F.relu(self.linear1(x))
        scores = self.linear2(l1scores)
        return scores

class ThreeLayerNet(nn.Module):
    def __init__(self, input_dim, num_classes, h1, h2):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, num_classes)
    
    def forward(self, x):
        l1scores = F.relu(self.linear1(x))
        l2scores = F.relu(self.linear2(l1scores))
        scores = self.linear3(l2scores)
        return scores
