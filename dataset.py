import numpy as np
import torch
from torch.utils import data


class CSVDataset(data.Dataset):
    def __init__(self, x_csv_file, y_csv_file):
        self.x_scv_file_name = x_csv_file
        self.y_scv_file_name = y_csv_file
        self.len = 0
        self.x_matrix = []
        self.y_matrix = []
        for line in open(self.x_scv_file_name):
            self.len += 1
            x = line.split(',')
            x = [float(i) / 255 for i in x]
            self.x_matrix.append(x)
        for label in open(self.y_scv_file_name):
            self.y_matrix.append(int(label))
        self.x_matrix = np.array(self.x_matrix)
        self.y_matrix = np.array(self.y_matrix)
        pass

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.Tensor(self.x_matrix[index])
        y = self.y_matrix[index]
        return x, y
