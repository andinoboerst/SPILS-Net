import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def scale_data(data, split_indices: list, concatenate=False):

    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Combine all training simulations into a single NumPy array (for fitting)
    combined_data = np.concatenate(data[split_indices[0]], axis=0)

    scaler.fit(combined_data)

    scaled_X = np.array([scaler.transform(x) for x in data])

    def concat_func(x):
        if concatenate:
            return np.concatenate(x, axis=0)
        else:
            return x

    train_data = concat_func(scaled_X[split_indices[0]])

    additional_data = []

    for sims in split_indices[1:]:
        additional_data.append(concat_func(scaled_X[sims]))

    return scaler, train_data, *additional_data


def train_val_test_split_tct():
    train_sims = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21]
    val_sims = [7, 13, 19]
    test_sims = [4, 10, 16]

    return train_sims, val_sims, test_sims


class SimulationDataset(Dataset):
    def __init__(self, *args, device: str | None = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data = [torch.tensor(d, dtype=torch.float64).to(device) for d in args]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)

    def append(self, *args) -> None:
        """
        Append new data to the existing dataset.
        """
        new_data_tensors = [torch.tensor(d, dtype=torch.float64).to(self.data[i].device) for i, d in enumerate(args)]
        self.data = [torch.cat((self.data[i], new_data_tensors[i])) for i in range(len(self.data))]
