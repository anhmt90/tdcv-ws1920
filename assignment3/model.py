from torch.nn import Sequential
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 7, 5)
        self.fc1 = nn.Linear(12 * 12 * 7, 256)
        self.fc2 = nn.Linear(256, 16)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 12 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
