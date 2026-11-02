"""xVGAE stage 1: a global latent z0 per graph and a latent z_i per node, and sampling with the z0 DDPM.

Generation: n ~ training sizes, z0 ~ p(z0 | n) (the z0 DDPM in stage 2), z_i ~ p(z_i | z0), A_ij ~ Bernoulli(σ(s_ij)).
Loss per batch of B graphs: [sum_ij BCE + beta sum_i KL(q(z_i) || p(z_i | z0)) + beta0 KL(q(z0) || p(z0 | n))] / B.
"""
import json
import os

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj

from xvgae.layers import SpectralEmbedding, reparameterize, tril_indices
from xvgae.encoder import LocalEncoder, GlobalEncoder, GTLocalEncoder, GTGlobalEncoder, HierarchicalEncoder
from xvgae.decoder import SetDecoder
from xvgae.prior import NormalPrior, NConditioner, SizeHead, z0_size_prior

# Architecture of the standard recipe. decoder "sab" = xVGAE-SAB, "mlp" = xVGAE-MLP; encoder_type "gt", size_head,
# cond_decoder_n, local_cond_source "z0" and prior_fixed_var_one are used by some paper rows (configs/).
ARCH = dict(z_dim=8, z0_dim=32, hidden_dim=64, num_layers=5, attn_heads=8, bilinear_heads=8, rank=4, decoder="sab",
            encoder_type="sab", local_cond_source="eigvals", cond_z0_prior_n=True, cond_decoder_n=False,
            prior_fixed_var_one=False, size_head=False)


class HierarchicalGraphLVAE(nn.Module):
    def __init__(self, spectral_embedding, encoder, prior, decoder, config):
        super().__init__()
        self.spectral_embedding = spectral_embedding
        self.encoder = encoder
        self.prior = prior
        self.decoder = decoder
        self.z0_dim, self.z_dim = int(config["z0_dim"]), int(config["z_dim"])
        self.sparse_encoder = config["encoder_type"] == "gt"
        self.size_head = SizeHead(self.z0_dim, config["hidden_dim"]) if config["size_head"] else None
        self.cond_decoder_n = bool(config["cond_decoder_n"])
        self.z0_n_prior = z0_size_prior(self.z0_dim) if config["cond_z0_prior_n"] else None
        self.n_conditioner = NConditioner() if (self.cond_decoder_n or config["cond_z0_prior_n"]) else None
        self.local_cond_source = config["local_cond_source"]
        self.eigvals_to_local_cond = nn.LazyLinear(self.z0_dim) if self.local_cond_source == "eigvals" else None

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch):
        """Encode, draw z0 and z from the posteriors, decode."""
        x, batch_or_mask = self.spectral_embedding(batch, return_sparse=self.sparse_encoder)
        edge_index = batch.edge_index if self.sparse_encoder else None
        mu0, lv0 = self.encoder.global_encode(x, batch_or_mask, batch.eigvals, edge_index=edge_index)
        z0 = reparameterize(mu0, lv0)
        cond = self.eigvals_to_local_cond(batch.eigvals) if self.local_cond_source == "eigvals" else z0
        mu_e, lv_e = self.encoder.local_encode(x, batch_or_mask, cond, edge_index=edge_index)
        if self.sparse_encoder:
            mu_e, mask = to_dense_batch(mu_e, batch.batch)
            lv_e, _ = to_dense_batch(lv_e, batch.batch)
            mu_e, lv_e = mu_e * mask.unsqueeze(-1), lv_e * mask.unsqueeze(-1)
        else:
            mask = batch_or_mask
        num_nodes = mask.sum(dim=-1).long()
        mu_p, lv_p = self.prior(z0)
        size_log_prob = self.size_head.log_prob(z0, num_nodes).mean() if self.size_head is not None else None
        n_emb = self.n_conditioner(num_nodes) if self.cond_decoder_n else None
        logits = self.decoder(reparameterize(mu_e, lv_e), z0, mask=mask, n_emb=n_emb)
        return dict(logits_vec=logits, z0=z0, mu0=mu0, lv0=lv0, mu_post=mu_e, lv_post=lv_e, mu_p=mu_p, lv_p=lv_p,
                    mask=mask, size_log_prob=size_log_prob)

    def z0_prior_params(self, n):
        """(mu, logvar) of p(z0 | n) for sizes n [B]."""
        x = torch.log(torch.as_tensor(n, dtype=torch.float32, device=self.device)).view(-1, 1)
        o = self.z0_n_prior(x)
        return o[:, :self.z0_dim], o[:, self.z0_dim:].clamp(-8.0, 4.0)

    def compute_loss(self, out, batch, beta=1.0, beta0=1.0, free_bits_z=None):
        """Negative ELBO in nats per graph (the node-latent KL floored at free_bits_z per node and dimension) + stats."""
        logits, mask = out["logits_vec"], out["mask"]
        B, N = mask.shape
        adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=N)
        adj = torch.clamp(adj + adj.transpose(1, 2), 0, 1)
        ti, tj = tril_indices(N, logits.device)
        pair_mask = mask[:, ti] & mask[:, tj]
        n_pairs, n_nodes = pair_mask.sum().clamp(min=1), mask.sum().clamp(min=1)
        recon_sum = (F.binary_cross_entropy_with_logits(logits, adj[:, ti, tj], reduction="none") * pair_mask).sum()
        mu0, lv0 = out["mu0"], out["lv0"]
        if self.z0_n_prior is not None:
            mp, lp = self.z0_prior_params(mask.sum(1))
            kl_z0_sum = (0.5 * (lp - lv0 + (lv0.exp() + (mu0 - mp).pow(2)) / lp.exp() - 1.0).sum(-1)).sum()
        else:
            kl_z0_sum = (-0.5 * (1 + lv0 - mu0.pow(2) - lv0.exp()).sum(-1)).sum()
        mu, lv, mp, lp = out["mu_post"], out["lv_post"], out["mu_p"], out["lv_p"]
        kl_node = 0.5 * (lp - lv + torch.exp(lv - lp) + (mu - mp).pow(2) * torch.exp(-lp) - 1.0).sum(-1)
        kl_z_sum = torch.where(mask.bool(), kl_node, torch.zeros_like(kl_node)).sum()
        kl_z_reg = kl_z_sum if free_bits_z is None else \
            torch.clamp_min(kl_z_sum, float(free_bits_z) * n_nodes.float() * self.z_dim)
        total = recon_sum + beta * kl_z_reg + beta0 * kl_z0_sum
        if out["size_log_prob"] is not None:
            total = total - out["size_log_prob"] * float(B)
        loss = total / float(B)
        stats = dict(loss=loss.item(), recon=(recon_sum / n_pairs).item(), kl_z=(kl_z_sum / (n_nodes * self.z_dim)).item(),
                     kl_z0=(kl_z0_sum / (float(B) * float(mu0.shape[-1]))).item())
        return loss, stats

    @torch.no_grad()
    def sample_logits(self, num_nodes, z0=None):
        """Edge logits of graphs of the given sizes, z ~ p(z | z0); z0 given, else drawn from p(z0 | n) or N(0, I)."""
        self.eval()
        dev, ns = self.device, list(num_nodes)
        G, N = len(ns), max(ns)
        if z0 is not None:
            z0 = z0.to(dev)
        elif self.z0_n_prior is not None:
            mp, lp = self.z0_prior_params(torch.tensor(ns, device=dev))
            z0 = mp + torch.exp(0.5 * lp) * torch.randn(G, self.z0_dim, device=dev)
        else:
            z0 = torch.randn(G, self.z0_dim, device=dev)
        mask = torch.zeros(G, N, dtype=torch.bool, device=dev)
        for i, n in enumerate(ns):
            mask[i, :n] = True
        n_emb = self.n_conditioner(torch.tensor(ns, device=dev)) if self.cond_decoder_n else None
        mu_p, lv_p = self.prior(z0)
        z = self.prior.sample(mu_p, lv_p, num_nodes=N, mask=mask)
        return dict(logits_vec=self.decoder(z, z0, mask=mask, n_emb=n_emb), mask=mask)

    @torch.no_grad()
    def encode_and_generate(self, loader):
        """Encoded z0, then z ~ p(z | z0), decoded at each graph's own n (the validation mode encz0gen)."""
        was_training = self.training
        self.eval()
        adjs = []
        for batch in loader:
            out = self.forward(batch.to(self.device))
            for g in range(out["mask"].shape[0]):
                n = int(out["mask"][g].sum().item())
                z0, mask = out["z0"][g:g + 1], torch.ones(1, n, dtype=torch.bool, device=self.device)
                mu_p, lv_p = self.prior(z0)
                n_emb = self.n_conditioner(torch.tensor([n], device=self.device)) if self.cond_decoder_n else None
                logits = self.decoder(self.prior.sample(mu_p, lv_p, num_nodes=n, mask=mask), z0, mask=mask, n_emb=n_emb)
                adjs.append(bernoulli_adj(logits.cpu(), n))
        self.train(was_training)
        return adjs


XVGAE = HierarchicalGraphLVAE


def bernoulli_adj(logits, n):
    """One Bernoulli graph from [1, E] lower-triangle logits -> symmetric (n, n) float64 array."""
    edges = torch.bernoulli(torch.sigmoid(logits.squeeze(0)))
    N = int((1 + np.sqrt(1 + 8 * edges.shape[0])) / 2)
    A = torch.zeros(N, N)
    ti, tj = tril_indices(N, torch.device("cpu"))
    A[ti, tj] = edges
    return (A + A.T)[:n, :n].numpy().astype(np.float64)


def build_model(cfg, train_loader, device="cpu"):
    """Stage 1 from a config; the first training batch fixes the input widths."""
    cfg = {**ARCH, **cfg}
    b = next(iter(train_loader))
    k, H, Z, Z0, L = b.eigvals.shape[1], cfg["hidden_dim"], cfg["z_dim"], cfg["z0_dim"], cfg["num_layers"]
    embed = SpectralEmbedding(b.x.shape[1], k, H)
    if cfg["encoder_type"] == "gt":
        heads = cfg["attn_heads"]
        enc = HierarchicalEncoder(GTGlobalEncoder(H, k, Z0, L, heads), GTLocalEncoder(H, Z, Z0, L, heads))
    else:
        enc = HierarchicalEncoder(GlobalEncoder(H, k, Z0, L), LocalEncoder(H, Z, Z0, L))
    prior = NormalPrior(Z0, Z, H, fixed_var=cfg["prior_fixed_var_one"])
    decoder = SetDecoder(Z, Z0, H, bilinear_heads=cfg["bilinear_heads"], attn_heads=cfg["attn_heads"], rank=cfg["rank"],
                         num_layers=L if cfg["decoder"] == "sab" else 0, cond_decoder_n=cfg["cond_decoder_n"],
                         n_emb_dim=32 if cfg["cond_decoder_n"] else 0)
    return HierarchicalGraphLVAE(embed, enc, prior, decoder, cfg).to(device)


def resolve_ckpt(run_dir, ckpt):
    """"auto" = the validation pick of scripts/select_stage1.py (<run>/stage1_pick.json); any other name is kept."""
    if ckpt != "auto":
        return ckpt
    pk = os.path.join(run_dir, "stage1_pick.json")
    if not os.path.exists(pk):
        raise FileNotFoundError(f"{pk}: run scripts/select_stage1.py --run {run_dir} first")
    return json.load(open(pk))["pick"]


def load_stage1(run_dir, ckpt="auto", device="cpu", shuffle_train=False, dataset=None):
    """(model, config, train, val, test loaders) of a stage-1 run; the checkpoint is loaded strict."""
    from xvgae.data import get_dataloaders
    cfg = {**ARCH, **json.load(open(os.path.join(run_dir, "config.json")))}
    cfg["dataset"] = dataset or cfg["dataset"]
    if cfg.get("_data_root") and "XVGAE_DATA_DIR" not in os.environ and os.path.isdir(cfg["_data_root"]):
        os.environ["XVGAE_DATA_DIR"] = cfg["_data_root"]
    tr, va, te = get_dataloaders(cfg["dataset"], cfg["batch_size"], shuffle_train=shuffle_train)
    m = build_model(cfg, tr, device)
    path = ckpt if os.path.dirname(ckpt) else os.path.join(run_dir, resolve_ckpt(run_dir, ckpt))
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return m.eval(), cfg, tr, va, te


def enforce_fp32():
    """No TF32 matmuls when sampling or scoring."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def edges_from_logits(logits, n):
    """One Bernoulli networkx graph from the n(n-1)/2 lower-triangle logits (entries past them are padding)."""
    ti, tj = torch.tril_indices(n, n, offset=-1, device=logits.device)
    p = torch.sigmoid(torch.nan_to_num(logits[:ti.shape[0]], nan=0., posinf=20, neginf=-20).clamp(-20, 20))
    e = torch.rand_like(p) < p
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(ti[e].cpu().numpy(), tj[e].cpu().numpy()))
    return G


@torch.no_grad()
def draw(model, loader, hist_n, seed, ddpm=None, device="cpu"):
    """One draw of len(loader.dataset) graphs: n from the training sizes, z0 from the DDPM (ddpm = (model, mu, sd)) or,
    without it, from p(z0 | n); the split fixes only the number of graphs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    graphs = []
    for batch in loader:
        ns = [int(x) for x in rng.choice(hist_n, size=batch.num_graphs)]
        z0 = None
        if ddpm is not None:
            dd, mu, sd = ddpm
            z0 = dd.sample(len(ns), device=device, sizes=torch.as_tensor(ns, device=device))
            z0 = z0 * sd.to(device) + mu.to(device)
        o = model.sample_logits(ns, z0=z0)
        for b in range(len(ns)):
            graphs.append(edges_from_logits(o["logits_vec"][b], ns[b]))
    return graphs
