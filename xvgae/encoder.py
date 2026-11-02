"""Encoders q(z0 | G) and q(z_i | G, c): set attention (the standard recipe) or GPS (the CiteSeer-Ego xVGAE-SAB row)."""
import torch.nn as nn
from torch_geometric.nn import GINConv, GPSConv, global_mean_pool

from xvgae.layers import SABPlusPlus, PMAPlusPlus, to_logvar


class LocalEncoder(nn.Module):
    """q(z_i | G, c): SAB layers FiLM-conditioned on c, then (mu_i, logvar_i)."""

    def __init__(self, hidden_dim, z_dim, z0_dim, num_layers=5, num_heads=1):
        super().__init__()
        self.layers = nn.ModuleList([SABPlusPlus(hidden_dim, num_heads, z0_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_lv = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, mask, cond):
        for layer in self.layers:
            x = layer(x, film_cond=cond, mask=mask)
        x = x + self.head(x)
        m = mask.unsqueeze(-1).float()
        return self.fc_mu(x) * m, to_logvar(self.fc_lv(x)) * m


class GlobalEncoder(nn.Module):
    """q(z0 | G): SAB layers FiLM-conditioned on the eigenvalues, PMA pooling, then (mu0, logvar0)."""

    def __init__(self, hidden_dim, k_spectral, z0_dim, num_layers=5, num_heads=1):
        super().__init__()
        self.layers = nn.ModuleList([SABPlusPlus(hidden_dim, num_heads, k_spectral) for _ in range(num_layers)])
        self.pool = PMAPlusPlus(hidden_dim, 1, num_heads, k_spectral)
        self.fc_mu = nn.Linear(hidden_dim, z0_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z0_dim)

    def forward(self, x, mask, eigvals):
        for layer in self.layers:
            x = layer(x, film_cond=eigvals, mask=mask)
        pooled = self.pool(x, film_cond=eigvals, mask=mask).squeeze(1)
        return self.fc_mu(pooled), to_logvar(self.fc_logvar(pooled))


def _gps_layer(hidden_dim, num_heads):
    gin = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, hidden_dim))
    return GPSConv(channels=hidden_dim, conv=GINConv(gin), heads=num_heads, dropout=0.0, act="relu", norm="batch_norm")


class GTLocalEncoder(nn.Module):
    """GPS node encoder (Rampasek et al. 2022) with FiLM on z0."""

    def __init__(self, hidden_dim, z_dim, z0_dim, num_layers=5, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.film_gen = nn.ModuleList([nn.Linear(z0_dim, 2 * hidden_dim) for _ in range(num_layers)])
        self.layers.extend([_gps_layer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_lv = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, batch_vec, z0, edge_index):
        for film, layer in zip(self.film_gen, self.layers):
            x = layer(x, edge_index, batch_vec)
            gamma, beta = film(z0).chunk(2, dim=-1)
            x = x * (1 + gamma[batch_vec]) + beta[batch_vec]
        x = x + self.head(x)
        return self.fc_mu(x), to_logvar(self.fc_lv(x))


class GTGlobalEncoder(nn.Module):
    """GPS graph encoder with FiLM on the eigenvalues and mean pooling."""

    def __init__(self, hidden_dim, k_spectral, z0_dim, num_layers=5, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.film_gen = nn.ModuleList([nn.Linear(k_spectral, 2 * hidden_dim) for _ in range(num_layers)])
        self.layers.extend([_gps_layer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fc_mu = nn.Linear(hidden_dim, z0_dim)
        self.fc_lv = nn.Linear(hidden_dim, z0_dim)

    def forward(self, x, batch_vec, eigvals, edge_index):
        for film, layer in zip(self.film_gen, self.layers):
            x = layer(x, edge_index, batch_vec)
            gamma, beta = film(eigvals).chunk(2, dim=-1)
            x = x * (1 + gamma[batch_vec]) + beta[batch_vec]
        pooled = global_mean_pool(x, batch_vec)
        pooled = pooled + self.pool(pooled)
        return self.fc_mu(pooled), to_logvar(self.fc_lv(pooled))


class HierarchicalEncoder(nn.Module):
    """The global and the local encoder."""

    def __init__(self, global_enc, local_enc):
        super().__init__()
        self.global_encoder = global_enc
        self.local_encoder = local_enc

    def global_encode(self, x, mask_or_batch, eigvals, edge_index=None):
        if isinstance(self.global_encoder, GTGlobalEncoder):
            return self.global_encoder(x, mask_or_batch, eigvals, edge_index)
        return self.global_encoder(x, mask_or_batch, eigvals)

    def local_encode(self, x, mask_or_batch, cond, edge_index=None):
        if isinstance(self.local_encoder, GTLocalEncoder):
            return self.local_encoder(x, mask_or_batch, cond, edge_index)
        return self.local_encoder(x, mask_or_batch, cond)
