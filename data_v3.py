import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
import numpy as np



class SpotifyDataset(Dataset):
    def __init__(self, filename, label):
        self.data = pd.read_csv(filename, header=None, usecols=range(10))
        # self.data.drop(columns=[10], inplace=True)
        self.data = self.data.fillna(0)
        self.data = self.data.astype('float64')
        self.data['class'] = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)
        return x, y

def load_spotify_data(filename, label):
    dataset = SpotifyDataset(filename, label)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader

def split_data_loader(loader):
    n = len(loader.dataset)
    train_size = int(n * 0.9)
    test_size = n - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(loader.dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_loader, test_loader


def combine_data_loaders(loader1, loader2):
    combined_dataset = ConcatDataset([loader1.dataset, loader2.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
    return combined_loader



def get_data_from_loader(loader):
    x_train, y_train = [], []
    for data, target in loader:
        x_train.append(data.numpy())
        y_train.append(target.numpy())

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return x_train, y_train

