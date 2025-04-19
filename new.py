import torch
import numpy as np
import pandas as pd
from model import construct_VAE
from components import construct_encoder, construct_decoder, construct_seasonal_prior
from config import INPUT_SIZE, LATENT_SIZE
import matplotlib.pyplot as plt

# Load the trained VAE model
def load_trained_model(model_path):
    # Load encoder, decoder, and seasonal prior
    encoder = construct_encoder()
    decoder = construct_decoder()
    seasonal_prior = construct_seasonal_prior()

    # Instantiate the VAE model
    vae_model = construct_VAE(encoder, decoder, seasonal_prior, input_size=INPUT_SIZE)
    vae_model.load_state_dict(torch.load(model_path))
    vae_model.eval()  # Set the model to evaluation mode
    return vae_model

# Function to generate synthetic data
def generate_synthetic_data(vae_model, start_time, end_time, seasonal_input_func, device='cpu'):
    # Generate a range of time steps (e.g., hourly data between start_time and end_time)
    time_range = pd.date_range(start=start_time, end=end_time, freq='H')
    n_samples = len(time_range)

    # Create seasonal input for each time step
    starting_day = np.array(time_range.dayofyear)[:, np.newaxis] - 1
    data_days = (starting_day + np.arange(0, n_samples // 24, LATENT_SIZE // 24)) % 365
    seasonal_data = seasonal_input_func(data_days / 365)

    # Convert seasonal input to tensor
    seasonal_tensor = torch.tensor(seasonal_data, dtype=torch.float32).to(device)

    # Generate synthetic temperature data
    synthetic_data = []
    with torch.no_grad():
        for i in range(0, n_samples, INPUT_SIZE):
            batch_seasonal = seasonal_tensor[i:i + INPUT_SIZE].unsqueeze(0)  # Shape (1, INPUT_SIZE, ...)
            z_mean, z_log_var, z = vae_model.encoder(batch_seasonal)
            reconstructed = vae_model.decoder(z)
            synthetic_data.extend(reconstructed.squeeze(0).cpu().numpy())

    # Convert to a DataFrame
    synthetic_data = np.array(synthetic_data).flatten()
    synthetic_df = pd.DataFrame({
        'timestamp': time_range[:len(synthetic_data)],
        'temperature': synthetic_data
    })

    return synthetic_df

# Save the generated synthetic data to CSV
def save_generated_data(synthetic_df, filename):
    synthetic_df.to_csv(filename, index=False)
    print(f"Generated data saved to {filename}")

if __name__ == "__main__":
    # Load the trained model
    model_path = 'models/vae_model_final.pth'
    vae_model = load_trained_model(model_path)

    # Generate synthetic data from the model
    start_time = '2025-01-01 00:00:00'
    end_time = '2025-01-10 23:59:59'

    seasonal_input_func = lambda x: np.stack([np.sin(2 * np.pi * i * x) for i in range(1, DEGREE + 1)] +
                                             [np.cos(2 * np.pi * i * x) for i in range(1, DEGREE + 1)], axis=-1)
    
    synthetic_data_df = generate_synthetic_data(vae_model, start_time, end_time, seasonal_input_func)

    # Save the synthetic data to CSV
    save_generated_data(synthetic_data_df, 'generated_temperature_data.csv')

    # Optional: Plot a sample of the generated data
    plt.figure(figsize=(10, 6))
    plt.plot(synthetic_data_df['timestamp'], synthetic_data_df['temperature'])
    plt.xlabel('Timestamp')
    plt.ylabel('Generated Temperature')
    plt.title('Generated Temperature Data over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
