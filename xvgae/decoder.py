"""The edge decoder p(A | z, z0): node representations h_i, then a chunked low-rank bilinear edge score.

s_ij = sum_h alpha_h(z0) [(g_h * h_i)^T W_h (g_h * h_j) + (h_i V_a P_a)(h_j V_a P_a)^T - (h_i V_r P_r)(h_j V_r P_r)^T]
       + b(z0) + bias(h_i) + bias(h_j),   W_h symmetric with unit spectral norm; g, P_a, P_r, alpha, b from z0.
xVGAE-SAB: h from num_layers SAB layers across nodes. xVGAE-MLP (num_layers = 0): h_i = f(z_i, z0), pointwise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from xvgae.layers import MLP, FiLM, SABPlusPlus, tril_indices


class SetDecoder(nn.Module):
    def __init__(self, z_dim, z0_dim, hidden_dim, bilinear_heads=8, attn_heads=8, rank=4, num_layers=5,
                 cond_decoder_n=False, n_emb_dim=0, logit_clamp=30.0):
        super().__init__()
        assert hidden_dim % bilinear_heads == 0, "hidden_dim must be divisible by bilinear_heads"
        self.bilinear_heads, self.rank, self.num_layers = bilinear_heads, rank, num_layers
        self.logit_clamp = float(logit_clamp)
        self.d_k = hidden_dim // bilinear_heads
        self.project_in = nn.Linear(z_dim, hidden_dim)
        self.cond_decoder_n = bool(cond_decoder_n)
        if self.cond_decoder_n:
            self.n_proj = MLP(n_emb_dim, [hidden_dim], hidden_dim)
            nn.init.zeros_(self.n_proj.layers[-1].weight)
            nn.init.zeros_(self.n_proj.layers[-1].bias)
        if num_layers > 0:
            self.refiner = nn.ModuleList([SABPlusPlus(hidden_dim, attn_heads, z0_dim) for _ in range(num_layers)])
        else:
            self.film = FiLM(hidden_dim, z0_dim)
            self.mlp = MLP(hidden_dim, [2 * hidden_dim, hidden_dim], hidden_dim)
        self.W_base_params = nn.Parameter(torch.randn(bilinear_heads, self.d_k * (self.d_k + 1) // 2) * 0.02)
        self.V_a = nn.Parameter(torch.randn(bilinear_heads, self.d_k, self.d_k) * 0.02)
        self.V_r = nn.Parameter(torch.randn(bilinear_heads, self.d_k, self.d_k) * 0.02)
        self.mod_dim = self.d_k * rank * 2
        self.node_bias = MLP(hidden_dim, [hidden_dim // 2], 1)
        self.gate_net = MLP(z0_dim, [hidden_dim], bilinear_heads * self.d_k)
        self.head_net = MLP(z0_dim, [hidden_dim], self.mod_dim * bilinear_heads + 1 + bilinear_heads)
        self._W_cache = (None, None)

    def W_base(self, device):
        """Symmetric base matrices [H, d_k, d_k], each divided by its (detached) spectral norm."""
        version = (self.W_base_params.data_ptr(), self.W_base_params._version)
        if self._W_cache[1] == version and not self.training and not torch.is_grad_enabled():
            return self._W_cache[0]
        W = torch.zeros(self.bilinear_heads, self.d_k, self.d_k, device=device, dtype=self.W_base_params.dtype)
        ti, tj = tril_indices(self.d_k, device, offset=0)
        W[:, ti, tj] = self.W_base_params
        W = W + W.transpose(-1, -2)
        d = torch.arange(self.d_k, device=device)
        W[:, d, d] /= 2.0
        W = W / torch.linalg.matrix_norm(W.detach(), ord=2).view(-1, 1, 1).clamp(min=1e-8)
        self._W_cache = (W, version)
        return W

    def forward(self, z, z0, mask=None, n_emb=None):
        """z [B, N, z_dim], z0 [B, z0_dim], mask [B, N] -> edge logits [B, N(N-1)/2] (lower triangle; padding -1e9)."""
        B, N, _ = z.shape
        H, dk = self.bilinear_heads, self.d_k
        h = self.project_in(z)
        if self.cond_decoder_n and n_emb is not None:
            h = h + self.n_proj(n_emb).unsqueeze(1)
        if self.num_layers > 0:
            for layer in self.refiner:
                h = layer(h, z0, mask=mask)
        else:
            h = self.film(h, z0, mask=mask)
            h = h + self.mlp(h, mask=mask)
        h_heads = h.view(B, N, H, dk).transpose(1, 2)                                   # [B, H, N, d_k]
        gamma = torch.sigmoid(self.gate_net(z0).view(B, H, dk)).unsqueeze(2)
        head_out = self.head_net(z0)
        mods = head_out[:, :H * self.mod_dim].view(B, H, self.mod_dim)
        g_bias = head_out[:, H * self.mod_dim:H * self.mod_dim + 1].view(B, 1, 1)
        alpha = F.softmax(head_out[:, -H:], dim=-1).view(B, H, 1, 1)
        h_gated = h_heads * gamma
        S_base = torch.matmul(torch.matmul(h_gated, self.W_base(z.device)), h_gated.transpose(-1, -2))
        pa = mods[:, :, :dk * self.rank].view(B, H, dk, self.rank)
        pr = mods[:, :, dk * self.rank:].view(B, H, dk, self.rank)
        h_pa = torch.einsum('bhne, bher -> bhnr', torch.einsum('bhnd, hde -> bhne', h_heads, self.V_a), pa)
        h_pr = torch.einsum('bhne, bher -> bhnr', torch.einsum('bhnd, hde -> bhne', h_heads, self.V_r), pr)
        S = S_base + (torch.einsum('bhnr, bhmr -> bhnm', h_pa, h_pa) - torch.einsum('bhnr, bhmr -> bhnm', h_pr, h_pr))
        bn = self.node_bias(h).squeeze(-1)
        adj = (S * alpha).sum(dim=1) + g_bias
        ti, tj = tril_indices(N, z.device)
        logits = (adj[:, ti, tj] + bn[:, ti] + bn[:, tj]).clamp(-self.logit_clamp, self.logit_clamp)
        if mask is not None:
            logits = logits.masked_fill(~(mask[:, ti] & mask[:, tj]), -1e9)
        return logits
