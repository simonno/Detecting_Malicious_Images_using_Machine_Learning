from __future__ import print_function
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import CSVDataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def data_loaders(dev_per=0.2, batch_size=8):
    csv_dataset = CSVDataset("x.csv", "y.csv")

    # train dev split
    indices = list(range(len(csv_dataset)))
    split = int(dev_per * len(csv_dataset))
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    train_loader = DataLoader(dataset=csv_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    validation_loader = DataLoader(dataset=csv_dataset, batch_size=len(validation_idx), sampler=validation_sampler,
                                   num_workers=2)
    return train_loader, validation_loader


net = Net()

