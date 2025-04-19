import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
from utils import *

# Load the full time series (hourly, 64 days per sequence)
time_series_df = pd.read_csv('data/final_timeseries.csv', index_col=None)

def flatten_time_series_into_chunks_of_data(dft):
    # 1536 columns = 64 days of hourly data
    k = INPUT_SIZE
    n = dft.shape[0] // k  # number of full k-length sequences
    dft_reshaped = pd.DataFrame(
        dft['t2m_norm_adj'].values[:n * k].reshape(n, k),
        index=dft['valid_time'].iloc[::k][:n].values,
        columns=range(1, k + 1)
    )
    return dft_reshaped

def fourier(x, degree=DEGREE):
    return np.stack(
        [np.sin(2 * np.pi * i * x) for i in range(1, degree + 1)] +
        [np.cos(2 * np.pi * i * x) for i in range(1, degree + 1)],
        axis=-1
    )

class ClimateDataset(Dataset):
    def __init__(self, csv_path, degree=DEGREE, input_size=INPUT_SIZE, latent_size=LATENT_SIZE):
        self.df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.degree = degree
        self.input_size = input_size
        self.latent_size = latent_size
        
        # Prepare input values
        self.values = self.df.values.astype(np.float32)

        # Prepare seasonal components using Fourier series
        starting_day = np.array(self.df.index.dayofyear)[:, np.newaxis] - 1
        data_days = (starting_day + np.arange(0, input_size // 24, latent_size // 24)) % 365
        self.seasonal_data = fourier(data_days / 365, degree).astype(np.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        # Return the sequence (value, seasonal) as a tuple
        return torch.tensor(self.values[idx], dtype=torch.float32), torch.tensor(self.seasonal_data[idx], dtype=torch.float32)

def get_dataloaders(batch_size=BATCH_SIZE, shuffle_train=True, training_ratio=0.8, csv_path='data/phoenix_64days.csv'):
    dataset = ClimateDataset(csv_path=csv_path)
    
    # Split data into train/test based on the training ratio
    n_train = int(len(dataset) * training_ratio)
    
    train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
    test_dataset = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader

if __name__ == "__main__":
    np.random.seed(42)
    dft_reshaped = flatten_time_series_into_chunks_of_data(time_series_df).sample(frac=1)
    dft_reshaped.to_csv(f'data/phoenix_{N_DAYS}days.csv')
    print(dft_reshaped.head())