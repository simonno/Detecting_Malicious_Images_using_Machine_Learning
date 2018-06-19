import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# NN without droptout and without batch normalization
from dataset import CSVDataset


class FirstNetWithoutNothing(nn.Module):
    def _init_(self, image_size):
        super(FirstNetWithoutNothing, self)._init_()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


# NN with dropout
class FirstNetWithDropout(nn.Module):
    def _init_(self, image_size):
        super(FirstNetWithDropout, self)._init_()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.235)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x)


# NN with batch normalization
class FirstNetWithBatchNorm(nn.Module):
    def _init_(self, image_size):
        super(FirstNetWithBatchNorm, self)._init_()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc0_bn = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
        self.fc2_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0_bn(self.fc0(x)))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x)


def train(model):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data = Variable(data)
        output = model(data)
        labels = torch.autograd.Variable(labels).long()
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(test_loader, name, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = Variable(data)
        output = model(data)
        target = torch.autograd.Variable(target).long()
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= test_loader.sampler._len_()
    print('\n ' + name + ': Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                      test_loader.sampler._len_(),
                                                                                      100. * correct /
                                                                                      test_loader.sampler._len_()))
    return test_loss


def data_loaders(dev_per=0.2, batch_size=8):
    csv_dataset = CSVDataset("x.csv", "y.csv")

    # train dev split
    indices = list(range(len(csv_dataset)))
    split = int(dev_per * len(csv_dataset))
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    train_loader = DataLoader(dataset=csv_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset=csv_dataset, batch_size=len(validation_idx), sampler=validation_sampler)
    return train_loader, validation_loader


train_loader, dev_loader = data_loaders()

lr = 0.001
# create 3 models
model = FirstNetWithoutNothing(32 * 32 * 3)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
arrTrainY = []
arrTestY = []
arrX = []
for epoch in range(1, 50 + 1):
    train(model)
    test_train = test(train_loader, "Train set", model)
    test_validation = test(dev_loader, "Dev set", model)
