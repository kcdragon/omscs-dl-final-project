import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class MNIST_Loader():
    def __init__(self, DIR, DATA_DIR=None):
        self.DIR = DIR
        if DATA_DIR:
            self.DATA_DIR = DATA_DIR
        else:
            self.DATA_DIR = f'{DIR}/data'
        # load full training set
        self.data = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, 
                                          transform=transforms.ToTensor())
        
    def train_val_split(self, test_size=0.1, shuffle=True):
        data = self.data
        all_indices = range(len(data))
        train_indices, val_indices, _, _ = train_test_split(
            all_indices,
            data.targets,
#             stratify=data.targets,
            test_size=test_size,
            shuffle=shuffle 
        )
        train_split = torch.utils.data.Subset(data, train_indices)
        val_split = torch.utils.data.Subset(data, val_indices)

#         print(f"{len(training_split)} in training set")
#         print(f"{len(validation_split)} in validation set")
        
        return train_split, val_split