import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()
        self.lin1  = nn.Linear(1, 128)
        self.lin2  = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    
class NotSoBasicNN(nn.Module):
    def __init__(self):
        super(NotSoBasicNN, self).__init__()
        self.lin1  = nn.Linear(1, 128)
        self.lin2  = nn.Linear(128, 256)
        self.lin3  = nn.Linear(256, 128)
        self.lin4  = nn.Linear(128, 2)

    def forward(self, x):
        x = F.silu(self.lin1(x))
        x = F.silu(self.lin2(x))
        x = F.silu(self.lin3(x))
        x = self.lin4(x)
        return x
    
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        return self.fc(x)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]