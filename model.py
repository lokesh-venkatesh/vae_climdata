import torch
import torch.nn as nn
from utils import *
from components import construct_encoder, construct_decoder, construct_seasonal_prior
from config import *

class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior, input_size, noise_log_var=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.input_size = input_size

        # Learnable noise log variance (if not provided)
        if noise_log_var is None:
            self.noise_log_var = nn.Parameter(torch.zeros(1))
        else:
            self.noise_log_var = noise_log_var

    def forward(self, values, seasonal):
        # Encoder pass
        z_mean, z_log_var, z = self.encoder(values)
        # Decoder pass
        reconstructed = self.decoder(z)
        # Seasonal prior pass
        seasonal_z_mean, seasonal_z_log_var, _ = self.prior(seasonal)

        # Losses
        recon_loss = log_lik_normal_sum(values, reconstructed, self.noise_log_var) / self.input_size
        kl_loss = kl_divergence_sum(z_mean, z_log_var, seasonal_z_mean, seasonal_z_log_var) / self.input_size

        return reconstructed, recon_loss, kl_loss

    def training_step(self, batch, optimizer):
        self.train()
        values, seasonal = batch
        optimizer.zero_grad()
        _, recon_loss, kl_loss = self(values, seasonal)
        total_loss = recon_loss + kl_loss
        total_loss.backward()
        optimizer.step()
        return {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def evaluation_step(self, batch):
        self.eval()
        with torch.no_grad():
            values, seasonal = batch
            _, recon_loss, kl_loss = self(values, seasonal)
            total_loss = recon_loss + kl_loss
            return {
                'loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            }


def construct_VAE(input_size=INPUT_SIZE, latent_size=LATENT_SIZE, DEGREE=DEGREE, 
                  interim_filters=64, latent_filter=32, noise_log_var=None):
    """
    Constructs and returns a VAE model with default parameters.
    """
    # Construct the Encoder, Decoder, and Seasonal Prior
    encoder = construct_encoder(input_shape=input_size, interim_filters=interim_filters, latent_filter=latent_filter)
    decoder = construct_decoder(latent_dim=latent_size, latent_filter=latent_filter, interim_filters=interim_filters)
    seasonal_prior = construct_seasonal_prior(latent_dim=latent_size, DEGREE=DEGREE, latent_filter=latent_filter)

    # Construct the VAE model
    vae = VAE(encoder=encoder, decoder=decoder, prior=seasonal_prior, 
              input_size=input_size, noise_log_var=noise_log_var)
    
    return vae
