"""Priors of stage 1: the node prior p(z_i | z0), the global prior p(z0 | n), the size embedding and the NB size head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial

from xvgae.layers import to_logvar


class _ResMLPBlock(nn.Module):
    """x -> x + Linear(SiLU(Linear(SiLU(x))))."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)


class _ResHead(nn.Module):
    """Linear -> n_blocks residual blocks -> Linear."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_blocks=3):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([_ResMLPBlock(hidden_dim) for _ in range(n_blocks)])
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h)
        return self.proj_out(h)


class NormalPrior(nn.Module):
    """p(z_i | z0) = N(mu(z0), diag sigma^2(z0)), the same for every node; N(0, I) at initialisation."""

    def __init__(self, z0_dim, z_dim, hidden_dim, fixed_var=False):
        super().__init__()
        self.z_dim, self.fixed_var = z_dim, bool(fixed_var)
        self.mu_net = _ResHead(z0_dim, hidden_dim, z_dim)
        self.lv_net = _ResHead(z0_dim, hidden_dim, z_dim)
        for net in (self.mu_net, self.lv_net):
            nn.init.zeros_(net.proj_out.weight)
            nn.init.zeros_(net.proj_out.bias)

    def forward(self, z0):
        """z0 [B, z0_dim] -> (mu, logvar), each [B, 1, z_dim]."""
        mu = self.mu_net(z0).view(z0.shape[0], 1, self.z_dim)
        lv = torch.zeros_like(mu) if self.fixed_var else to_logvar(self.lv_net(z0).view(z0.shape[0], 1, self.z_dim))
        return mu, lv

    def sample(self, mu, lv, num_nodes, mask=None):
        """z [B, N, z_dim], every node drawn from its graph's prior; padded nodes zeroed."""
        mu, lv = mu.expand(-1, num_nodes, -1), lv.expand(-1, num_nodes, -1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return z * mask.unsqueeze(-1).float() if mask is not None else z


def z0_size_prior(z0_dim, hidden=64):
    """The network of p(z0 | n): log n -> (mu, logvar); N(0, I) at initialisation."""
    net = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * z0_dim))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class NConditioner(nn.Module):
    """Learned embedding of the graph size n in [0, n_max]."""

    def __init__(self, n_max=2048, n_embed_dim=32):
        super().__init__()
        self.n_max = int(n_max)
        self.embedding = nn.Embedding(self.n_max + 1, n_embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, n):
        if bool((n < 0).any()) or bool((n > self.n_max).any()):
            raise ValueError(f"graph size outside [0, {self.n_max}]")
        return self.embedding(n.long())


class SizeHead(nn.Module):
    """Negative-binomial size head p(n | z0) (a training term of the CiteSeer-Ego xVGAE-SAB row)."""

    def __init__(self, z0_dim, hidden_dim, n_min=2, n_max=512):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(z0_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.readout = nn.Linear(hidden_dim, 2)
        with torch.no_grad():
            nn.init.normal_(self.readout.weight, mean=0.0, std=1e-3)
            self.readout.bias.copy_(torch.tensor([0.5 * (n_min + n_max), 0.0]))

    def log_prob(self, z0, n):
        out = self.readout(self.backbone(z0))
        r = F.softplus(out[..., 0]) + 1e-4
        p = torch.sigmoid(out[..., 1]).clamp(1e-4, 1.0 - 1e-4)
        return NegativeBinomial(total_count=r, probs=p).log_prob(n.float())
