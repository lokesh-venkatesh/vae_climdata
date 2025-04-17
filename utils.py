import torch
import torch.nn as nn
import numpy as np
import math

class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z"""
    def forward(self, z_mean, z_log_var):
        batch = z_mean.size(0)
        rank = z_mean.dim()

        if rank == 2:  # 2D case
            dim = z_mean.size(1)
            epsilon_shape = (batch, dim)
        elif rank == 1:  # 1D case
            epsilon_shape = (batch,)
        elif rank == 3:  # 3D case
            dim1 = z_mean.size(1)
            dim2 = z_mean.size(2)
            epsilon_shape = (batch, dim1, dim2)
        else:
            raise ValueError("z_mean and z_log_var must be 1D, 2D, or 3D tensors")

        epsilon = torch.randn(epsilon_shape, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


def kl_divergence_sum(mu1=0.0, log_var1=0.0, mu2=0.0, log_var2=0.0):
    var1 = torch.exp(log_var1)
    var2 = torch.exp(log_var2)
    axis0 = 0.5 * torch.mean(log_var2 - log_var1 + (var1 + (mu1 - mu2) ** 2) / var2 - 1, dim=0)
    return torch.sum(axis0)


def log_lik_normal_sum(x, mu=0.0, log_var=0.0):
    axis0 = -0.5 * (math.log(2 * np.pi) + torch.mean(log_var + ((x - mu) ** 2) * torch.exp(-log_var), dim=0))
    return torch.sum(axis0)
