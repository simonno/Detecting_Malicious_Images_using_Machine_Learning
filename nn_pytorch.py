from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import CSVDataset


class Net(nn.Module):
    def __init__(self, image_size):
        super(Net, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 3000)
        self.fc1 = nn.Linear(3000, 3000)
        self.fc2 = nn.Linear(3000, 3000)
        self.fc3 = nn.Linear(3000, 2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class FirstNet(Net):
    def __init__(self, image_size):
        super(FirstNet, self).__init__(image_size)
        self.bn0 = nn.BatchNorm1d(200)
        self.bn1 = nn.BatchNorm1d(200)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SecondNet(Net):
    def __init__(self, image_size):
        super(SecondNet, self).__init__(image_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, train_loader, epoch, device):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.sampler),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return model


def test(type_of_set, model, loader, size):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= size
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(type_of_set, test_loss, correct, size,
                                                                           100. * correct / size))
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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = data_loaders()
    size_train = len(train_loader.sampler)
    size_validation = len(validation_loader.sampler)

    lr = 0.001
    image_size = 32 * 32 * 3
    model = Net(image_size=image_size)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.AdaDelta(model.parameters(), lr=lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)

    losses_train = []
    losses_validation = []
    for epoch in range(1, 50 + 1):
        print('epoch: ' + str(epoch))
        model = train(model, optimizer, train_loader, epoch, device)
        losses_train.append(test('train', model, train_loader, size_train))
        losses_validation.append(test('validation', model, validation_loader, size_validation))
        print()


if __name__ == '__main__':
    main()
