import torch
import torch.nn as nn
import numpy as np
import random
import itertools
from torch.utils.data import IterableDataset

class C4ValidationIterableDataset(IterableDataset):
    def __init__(self, data_source, split='validation', valid_split=0.5, seed=None):
        self.data_source = data_source
        self.valid_split = valid_split
        self.split = split
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)
        data_iter = iter(self.data_source)
        total_size = 1000
        valid_size = int(total_size * self.valid_split)
        if self.split == 'validation':
            return itertools.islice(data_iter, 0, valid_size)
        else:
            return itertools.islice(data_iter, valid_size, total_size)

class TrainValTestIterableDataset(IterableDataset):
    def __init__(self, data_source, split='train', train_split=0.8, valid_split=0.1, seed=None):
        self.data_source = data_source
        self.split = split
        self.train_split = train_split
        self.valid_split = valid_split
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)
        data_iter = iter(self.data_source)

        total_size = 8000 
        train_size = int(total_size * self.train_split)
        valid_size = int(total_size * self.valid_split)

        if self.split == 'validation':
            return itertools.islice(data_iter, 0, valid_size)
        elif self.split == 'train':
            return itertools.islice(data_iter, valid_size, train_size + valid_size)
        elif self.split == 'test':
            return itertools.islice(data_iter, train_size + valid_size, total_size)

class CustomIterableDataset(IterableDataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __iter__(self):
        for sample in self.hf_dataset:
            yield sample

class DeltaNetwork(nn.Module):
    def __init__(self, input_dim=2048, layers=2, init_val=2.0):
        super(DeltaNetwork, self).__init__()
        if layers == 2:
            self.delta = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
            )
        elif layers == 3:
            self.delta = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1)
            )
        elif layers == 5:
            self.delta = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 1)
            )

        for layer in self.delta:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.init_val = init_val
        nn.init.constant_(self.delta[-1].bias, init_val)  # Set bias to the calculated value
    def forward(self, x):
        return self.delta(x)

class GammaNetwork(nn.Module):
    def __init__(self, input_dim=2048, layers=2, init_val=0.25):
        super(GammaNetwork, self).__init__()
        if layers == 2:
            self.gamma = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        elif layers == 3:
            self.gamma = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif layers == 5:
            self.gamma = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        for layer in self.gamma:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.init_val = init_val
        nn.init.constant_(self.gamma[-2].bias, np.log(init_val / (1 - init_val)))  # Set bias to the calculated value
    def forward(self, x):
        return self.gamma(x)
