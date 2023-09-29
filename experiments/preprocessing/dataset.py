"""
dataset.py

Core class definition for datasets, data preprocessing, and other necessary startup steps to run before
training the CAE Latent Action Model.
"""
from torch.utils.data import Dataset
import numpy as np


def get_dataset(n):
    return DemoTrainDataset(n), DemoTestDataset(n)


class DemoTrainDataset(Dataset):
    def __init__(self, n) -> None:
        super().__init__()
        self.size = n
        train_timesteps = np.linspace(-1, 1, self.size) * np.pi / 2
        self.xtrain = np.sin(train_timesteps)
        self.ytrain = np.cos(train_timesteps)

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> any:
        return {
            "state": self.xtrain[index],
            "action": self.ytrain[index],
        }


class DemoTestDataset(Dataset):
    def __init__(self, n) -> None:
        super().__init__()
        self.size = n
        test_timesteps = np.linspace(-1, 1, self.size // 5) * np.pi / 2
        self.xtest = np.sin(test_timesteps)
        self.ytest = np.cos(test_timesteps)

    def __len__(self):
        return self.size // 5

    def __getitem__(self, index) -> any:
        return {
            "state": self.xtest[index],
            "action": self.ytest[index],
        }
