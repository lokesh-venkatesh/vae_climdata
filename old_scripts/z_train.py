import numpy as np
import pandas as pd
from z_config import *
from z_utils import *
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
from z_model import VAE, Decoder, Encoder, SeasonalPrior  # assumes you have a model.py with a VAE class

data = pd.read_csv('data/phoenix_64days.csv', index_col=0, parse_dates=True)

fourier = lambda x: np.stack([np.sin(2*np.pi*i*x) for i in range(1, DEGREE+1)] + [np.cos(2*np.pi*i*x) for i in range(1, DEGREE+1)], axis=-1)

starting_day = np.array(data.index.dayofyear)[:, np.newaxis] - 1
data_days = (starting_day + np.arange(0, INPUT_SIZE//24, LATENT_SIZE//24)) % 365
seasonal_data = fourier(data_days/365)
print(seasonal_data.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming data and seasonal_data are NumPy arrays or pandas DataFrames
data_array = data.values if hasattr(data, "values") else data
seasonal_array = seasonal_data if isinstance(seasonal_data, np.ndarray) else seasonal_data.values

split_idx = int(len(data_array)*training_ratio)

# === Train/test split ===
train = data_array[:split_idx]
test = data_array[split_idx:]
train_seasonal = seasonal_array[:split_idx]
test_seasonal = seasonal_array[split_idx:]

# === Convert to torch tensors ===
train_tensor = torch.tensor(train, dtype=torch.float32)
test_tensor = torch.tensor(test, dtype=torch.float32)
train_seasonal_tensor = torch.tensor(train_seasonal, dtype=torch.float32)
test_seasonal_tensor = torch.tensor(test_seasonal, dtype=torch.float32)

# Move data to device
train_dataset = TensorDataset(train_tensor, train_seasonal_tensor)
test_dataset = TensorDataset(test_tensor, test_seasonal_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# === Initialize model and optimizer ===
vae_encoder = Encoder()
vae_decoder = Decoder()
vae_prior = SeasonalPrior()

vae = VAE(vae_encoder, vae_decoder, vae_prior).to(device)
optimizer = Adam(vae.parameters(), lr=learning_rate)

# === Training function ===
def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_data, batch_seasonal in dataloader:
        batch_data = batch_data.to(device)
        batch_seasonal = batch_seasonal.to(device)

        optimizer.zero_grad()
        recon, z_mean, z_log_var = model(batch_data, batch_seasonal)
        loss = model.loss_function(recon, batch_data, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Train Loss = {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader)


# === Evaluation function ===
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data, batch_seasonal in dataloader:
            batch_data = batch_data.to(device)
            batch_seasonal = batch_seasonal.to(device)

            recon, z_mean, z_log_var, _, _ = model(batch_data, batch_seasonal)
            loss = model.loss_function(recon, batch_data, z_mean, z_log_var)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(1, epochs + 1), desc="Training Progress"):
        train_loss = train_epoch(vae, train_loader, optimizer, epoch)
        val_loss = evaluate(vae, test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

    # === Optionally save model and losses ===
    torch.save(vae.state_dict(), "results/vae_model.pth")
    torch.save({
        "train_losses": train_losses,
        "val_losses": val_losses
    }, "training_history.pth")
    # === Generate and save training report ===
    report = pd.DataFrame({
        "Epoch": list(range(1, epochs + 1)),
        "Train Loss": train_losses,
        "Validation Loss": val_losses
    })

    report.to_csv("results/training_report.csv", index=False)
    print("Training report saved as 'training_report.csv'")

if __name__=="__main__":
    main()