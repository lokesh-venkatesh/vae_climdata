import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from utils import *

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, interim_filters, 5, stride=3, padding=2)
        self.conv2 = nn.Conv1d(interim_filters, interim_filters, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(interim_filters, interim_filters, 3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(interim_filters, interim_filters, 3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(interim_filters, interim_filters, 3, stride=2, padding=1)
        self.conv6 = nn.Conv1d(interim_filters, 2 * latent_filter, 3, stride=2, padding=1)
        self.sampling = Sampling()

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)  # [B, 2*latent_filter, T']
        x = x.permute(0, 2, 1)  # [B, T', 2*latent_filter]
        z_mean = x[:, :, :latent_filter]
        z_log_var = x[:, :, latent_filter:]
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(latent_filter, interim_filters, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(interim_filters, interim_filters, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(interim_filters, interim_filters, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose1d(interim_filters, interim_filters, 3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose1d(interim_filters, interim_filters, 3, stride=2, padding=1, output_padding=1)
        self.deconv6 = nn.ConvTranspose1d(interim_filters, 1, 5, stride=3, padding=2, output_padding=1)

    def forward(self, z):
        z = z.permute(0, 2, 1)  # [B, latent_filter, T]
        x = F.relu(self.deconv1(z))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = self.deconv6(x)
        return x.squeeze(1)  # [B, T]

class SeasonalPrior(nn.Module):
    def __init__(self):
        super(SeasonalPrior, self).__init__()
        self.fc = nn.Linear(2 * DEGREE, 2 * latent_filter, bias=False)
        self.sampling = Sampling()

    def forward(self, seasonal_input):
        x = self.fc(seasonal_input)
        x = x.view(-1, latent_dim, 2 * latent_filter)
        z_mean = x[:, :, :latent_filter]
        z_log_var = x[:, :, latent_filter:]
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

def kl_divergence_sum(mean1, log_var1, mean2, log_var2):
    return 0.5 * torch.sum(
        log_var2 - log_var1 +
        (torch.exp(log_var1) + (mean1 - mean2) ** 2) / torch.exp(log_var2) - 1
    )

def log_lik_normal_sum(x, mean, log_var):
    return -0.5 * torch.sum(torch.log(2 * torch.pi) + log_var + ((x - mean) ** 2) / torch.exp(log_var))

class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.noise_log_var = nn.Parameter(torch.zeros(1))

    def forward(self, values, seasonal):
        z_mean, z_log_var, z = self.encoder(values)
        reconstructed = self.decoder(z)
        seasonal_z_mean, seasonal_z_log_var, _ = self.prior(seasonal)
        return reconstructed, z_mean, z_log_var, seasonal_z_mean, seasonal_z_log_var

    def compute_loss(self, values, seasonal):
        reconstructed, z_mean, z_log_var, seasonal_z_mean, seasonal_z_log_var = self(values, seasonal)
        recon_loss = -log_lik_normal_sum(values, reconstructed, self.noise_log_var) / INPUT_SIZE
        kl_loss = kl_divergence_sum(z_mean, z_log_var, seasonal_z_mean, seasonal_z_log_var) / INPUT_SIZE
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss