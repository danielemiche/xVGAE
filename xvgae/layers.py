"""Building blocks: spectral node embedding, MLP, FiLM, SetNorm and set attention (SAB, PMA)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

VAR_EPS = 1e-4
_TRIL = {}


def tril_indices(n, device, offset=-1):
    """torch.tril_indices(n, n, offset), cached."""
    key = (n, str(device), offset)
    if key not in _TRIL:
        _TRIL[key] = torch.tril_indices(n, n, offset=offset, device=device)
    return _TRIL[key]


def reparameterize(mu, logvar):
    """mu + eps * exp(logvar / 2), eps ~ N(0, I), logvar clamped to [-10, 10]."""
    std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
    return mu + torch.randn_like(std) * std


def to_logvar(raw):
    """Head output -> log(softplus(raw) + eps)."""
    return torch.log(F.softplus(raw) + VAR_EPS)


class SpectralEmbedding(nn.Module):
    """Encoder input: [x ; eigvecs] -> 2-layer GCN -> concat with the input -> Linear."""

    def __init__(self, node_in_dim, k_spectral, hidden_dim):
        super().__init__()
        gcn_in_dim = k_spectral + max(node_in_dim, 0)
        self.gcn1 = GCNConv(gcn_in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.project = nn.Linear(gcn_in_dim + hidden_dim, hidden_dim)

    def forward(self, data, return_sparse=False):
        x = torch.cat([data.x, data.eigvecs], dim=-1)
        pe = self.act(self.gcn2(self.act(self.gcn1(x, data.edge_index)), data.edge_index))
        h = self.project(torch.cat([x, pe], dim=-1))
        return (h, data.batch) if return_sparse else to_dense_batch(h, data.batch)


class MLP(nn.Module):
    """Linear layers with SiLU in between (nn.Identity placeholders keep the parameter names stable)."""

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        dims = [input_size] + list(hidden_sizes) + [output_size]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(nn.Identity())

    def forward(self, x, mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Linear) and i < len(self.layers) - 1:
                x = F.silu(x)
        if mask is not None and x.dim() == 3:
            x = x * mask.unsqueeze(-1).float()
        return x


class FiLM(nn.Module):
    """x -> (1 + gamma(c)) x + delta(c); the identity at initialisation."""

    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.net = MLP(cond_dim, [2 * cond_dim], feature_dim * 2)
        nn.init.zeros_(self.net.layers[-1].weight)
        nn.init.zeros_(self.net.layers[-1].bias)

    def forward(self, x, cond, mask=None):
        gamma, delta = torch.chunk(self.net(cond), 2, dim=-1)
        out = (1.0 + gamma.unsqueeze(1)) * x + delta.unsqueeze(1)
        return out * mask.unsqueeze(-1).float() if mask is not None else out


class SetNorm(nn.Module):
    """Normalisation over the nodes and features of each graph, padded nodes excluded (Zhang et al. 2022)."""

    def __init__(self, feature_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        m = mask.unsqueeze(-1).float()
        count = torch.clamp(m.sum(dim=[1, 2], keepdim=True) * x.shape[-1], min=1.0)
        mean = (x * m).sum(dim=[1, 2], keepdim=True) / count
        var = (((x - mean) ** 2) * m).sum(dim=[1, 2], keepdim=True) / count
        return ((x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias) * m


class MABPlusPlus(nn.Module):
    """Multihead attention block: pre-norm (SetNorm), FiLM on both branches, residual feed-forward."""

    def __init__(self, d, num_heads, cond_dim):
        super().__init__()
        self.norm_x, self.norm_y, self.norm_ffn = SetNorm(d), SetNorm(d), SetNorm(d)
        self.film_x, self.film_y, self.film_ffn = FiLM(d, cond_dim), FiLM(d, cond_dim), FiLM(d, cond_dim)
        self.att = nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=0.0)
        self.rff = MLP(d, [2 * d], d)

    def forward(self, x, y, film_cond, mask_x=None, mask_y=None):
        q = self.film_x(self.norm_x(x, mask_x), film_cond, mask_x)
        k = self.film_y(self.norm_y(y, mask_y), film_cond, mask_y)
        h = x + self.att(q, k, k, key_padding_mask=~mask_y if mask_y is not None else None, need_weights=False)[0]
        out = h + self.rff(self.film_ffn(self.norm_ffn(h, mask_x), film_cond, mask_x), mask_x)
        return out * mask_x.unsqueeze(-1).float() if mask_x is not None else out


class SABPlusPlus(nn.Module):
    """Set self-attention block (permutation equivariant)."""

    def __init__(self, d, num_heads, cond_dim):
        super().__init__()
        self.mab = MABPlusPlus(d, num_heads, cond_dim)

    def forward(self, x, film_cond, mask=None):
        return self.mab(x, x, film_cond, mask_x=mask, mask_y=mask)


class PMAPlusPlus(nn.Module):
    """Pooling by multihead attention onto k learned seeds (permutation invariant)."""

    def __init__(self, d, k, num_heads, cond_dim):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, k, d))
        self.mab = MABPlusPlus(d, num_heads, cond_dim)
        self.rff_pool = MLP(d, [2 * d], d)

    def forward(self, x, film_cond, mask=None):
        res = self.mab(self.seed.expand(x.shape[0], -1, -1), x, film_cond, mask_x=None, mask_y=mask)
        return res + self.rff_pool(res)
