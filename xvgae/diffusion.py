"""Stage 2: an n-conditioned DDPM on the global latent z0 (eps-prediction MLP, 1000 steps, cosine schedule)."""
import glob
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xvgae import atomic_save, file_sha1, pick_device, write_json
from xvgae.layers import reparameterize


def make_schedule(n_steps=1000, s=0.008):
    """alpha_bar_t for t = 0..n_steps (Nichol & Dhariwal 2021)."""
    t = torch.linspace(0, 1, n_steps + 1)
    ab = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    return (ab / ab[0]).clamp(min=1e-6, max=1.0 - 1e-6)


def sinusoidal(t, dim):
    half = dim // 2
    emb = t.float()[:, None] * torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / half))[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Z0DDPM(nn.Module):
    """eps(z_t, t, n): an MLP with FiLM blocks on the time embedding plus an embedding of the graph size n."""

    def __init__(self, z0_dim, hidden=256, n_layers=3, n_steps=1000, t_emb_dim=64):
        super().__init__()
        self.z0_dim, self.n_steps, self.t_emb_dim = z0_dim, n_steps, t_emb_dim
        self.n_mlp = nn.Sequential(nn.Linear(t_emb_dim, t_emb_dim), nn.SiLU(), nn.Linear(t_emb_dim, t_emb_dim))
        nn.init.zeros_(self.n_mlp[-1].weight)
        nn.init.zeros_(self.n_mlp[-1].bias)
        self.in_proj = nn.Linear(z0_dim, hidden)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "norm": nn.LayerNorm(hidden),
            "mlp": nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.SiLU(), nn.Linear(4 * hidden, hidden)),
            "film": nn.Linear(t_emb_dim, 2 * hidden)}) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, z0_dim)
        self.register_buffer("alpha_bar", make_schedule(n_steps), persistent=False)

    def forward(self, z_t, t, n):
        n = torch.as_tensor(n, device=z_t.device).long()
        emb = sinusoidal(t, self.t_emb_dim) + self.n_mlp(sinusoidal(n, self.t_emb_dim))
        h = self.in_proj(z_t)
        for blk in self.blocks:
            gamma, beta = blk["film"](emb).chunk(2, dim=-1)
            h = h + blk["mlp"](blk["norm"](h) * (1.0 + gamma) + beta)
        return self.out_proj(self.out_norm(h))

    def loss(self, z0, n):
        t = torch.randint(1, self.n_steps + 1, (z0.shape[0],), device=z0.device)
        ab = self.alpha_bar[t].unsqueeze(-1)
        eps = torch.randn_like(z0)
        return F.mse_loss(self.forward(ab.sqrt() * z0 + (1.0 - ab).sqrt() * eps, t, n), eps)

    @torch.no_grad()
    def sample(self, n, device="cpu", sizes=None):
        """n standardised z0 draws (ancestral sampling) for graphs of the given sizes."""
        z = torch.randn(n, self.z0_dim, device=device)
        for t_int in range(self.n_steps, 0, -1):
            t = torch.full((n,), t_int, device=device, dtype=torch.long)
            ab_t, ab_tm = self.alpha_bar[t_int], self.alpha_bar[t_int - 1]
            alpha_t = ab_t / ab_tm
            z0_hat = (z - (1.0 - ab_t).sqrt() * self.forward(z, t, sizes)) / ab_t.sqrt().clamp_min(1e-6)
            mean = ab_tm.sqrt() * (1.0 - alpha_t) / (1.0 - ab_t).clamp_min(1e-6) * z0_hat \
                + alpha_t.sqrt() * (1.0 - ab_tm) / (1.0 - ab_t).clamp_min(1e-6) * z
            if t_int > 1:
                z = mean + ((1.0 - ab_tm) / (1.0 - ab_t).clamp_min(1e-6) * (1.0 - alpha_t)).sqrt() * torch.randn_like(z)
            else:
                z = mean
        return z


def load_z0_ddpm(ddpm_dir, z0_dim, device="cpu", state="z0_ddpm.pt"):
    """(ddpm, z0_mu, z0_sd) of a DDPM directory; state = z0_ddpm.pt or a snapshot z0_ddpm_ep<E>.pt / z0_ddpm_bestval.pt."""
    ld = lambda f: torch.load(os.path.join(ddpm_dir, f), map_location=device, weights_only=True)  # noqa: E731
    dd = Z0DDPM(z0_dim)
    dd.load_state_dict(ld(state), strict=True)
    return dd.to(device).eval(), ld("z0_mu.pt"), ld("z0_sd.pt")


@torch.no_grad()
def encode_pool(model, loader, reps=20, device="cpu"):
    """reps draws from q(z0 | G) per graph (rep-major rows) and the matching sizes n."""
    mus, lvs, ns = [], [], []
    for b in loader:
        out = model(b.to(device))
        mus.append(out["mu0"].cpu())
        lvs.append(out["lv0"].cpu())
        ns += [int(x) for x in out["mask"].sum(1).tolist()]
    mu, lv = torch.cat(mus, 0), torch.cat(lvs, 0)
    return torch.cat([reparameterize(mu, lv) for _ in range(reps)], 0), ns * reps


def val_eps_loss(model, zv, nv, device, batch_size=1024, seed=0):
    """Held-out eps-loss with seeded noise and steps (comparable across states)."""
    model.eval()
    tot = 0.0
    with torch.no_grad(), torch.random.fork_rng(devices=[device] if str(device).startswith("cuda") else []):
        torch.manual_seed(seed)
        for k in range(0, zv.shape[0], batch_size):
            b = zv[k:k + batch_size].to(device)
            tot += float(model.loss(b, nv[k:k + batch_size].to(device))) * b.shape[0]
    model.train()
    return tot / max(zv.shape[0], 1)


def train_z0_ddpm(model, pool, n_pool, epochs, batch_size=64, lr=2e-4, warmup_frac=0.05, weight_decay=1e-5, grad_clip=1.0,
                  out_dir=None, val_pool=None, val_n=None, snapshots=0, device="cpu"):
    """AdamW, linear warm-up then cosine LR, clip 1; writes z0_ddpm_ep<E>.pt (snapshots) and z0_ddpm_bestval.pt."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    G = pool.shape[0]
    steps_per_ep = max(1, math.ceil(G / batch_size))
    total = epochs * steps_per_ep
    warm = max(1, int(warmup_frac * total))
    snap = {int(round(epochs * (i + 1) / snapshots)) for i in range(snapshots)} if snapshots else set()
    every = max(1, epochs // 25)
    losses, vlog, step, best, t0, t_log = [], [], 0, float("inf"), time.time(), time.time()
    print(f"[z0-ddpm] {epochs} epochs x {steps_per_ep} steps (batch {batch_size})", flush=True)
    for ep in range(epochs):
        idx = torch.randperm(G)
        ep_loss = 0.0
        for k in range(0, G, batch_size):
            for g in opt.param_groups:
                g["lr"] = lr * step / warm if step < warm else \
                    0.5 * lr * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            loss = model.loss(pool[idx[k:k + batch_size]].to(device), n_pool[idx[k:k + batch_size]].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item()
            step += 1
        losses.append(ep_loss / steps_per_ep)
        msg = ""
        if val_pool is not None and ((ep + 1) % every == 0 or ep + 1 == epochs):
            v = val_eps_loss(model, val_pool, val_n, device)
            vlog.append((ep + 1, v))
            msg = f" val={v:.5f}"
            if v < best and out_dir:
                best = v
                atomic_save(model.state_dict(), os.path.join(out_dir, "z0_ddpm_bestval.pt"))
        if ep + 1 in snap and out_dir:
            atomic_save(model.state_dict(), os.path.join(out_dir, f"z0_ddpm_ep{ep + 1}.pt"))
        now = time.time()
        if (ep + 1) % every == 0 or now - t_log >= 60 or ep + 1 == epochs:
            t_log, el = now, now - t0
            print(f"[z0-ddpm] ep {ep + 1}/{epochs} loss={losses[-1]:.5f}{msg}  {el / 60:.1f} min, "
                  f"{1000 * el / step:.0f} ms/step, ~{el / step * (total - step) / 60:.0f} min left", flush=True)
    return losses, vlog


def fit(run_dir, ckpt, out_dir, epochs=2000, reps=20, lr=2e-4, batch_size=64, snapshots=2, seed=0, device=None,
        dataset=None):
    """Train the z0 DDPM of one stage-1 checkpoint on 20 posterior z0 draws per training graph (validation pool held
    out). Writes z0_ddpm.pt (last), z0_ddpm_ep<E>.pt, z0_ddpm_bestval.pt, z0_mu.pt, z0_sd.pt and ddpm_z0_meta.json."""
    from xvgae.model import load_stage1, resolve_ckpt
    ckpt = resolve_ckpt(run_dir, ckpt)          # "auto" -> the select_stage1.py pick, recorded by name in the meta
    dev = pick_device(device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    s1 = ckpt if os.path.dirname(ckpt) else os.path.join(os.path.abspath(run_dir), ckpt)
    sha1 = file_sha1(s1)                        # the file as loaded now, not as it may be after hours of training
    stage1, cfg, tr, va, _ = load_stage1(run_dir, s1, device=dev, shuffle_train=True, dataset=dataset)
    os.makedirs(out_dir, exist_ok=True)
    stale = ["ddpm_z0_meta.json", "state_pick.json", "z0_mu.pt", "z0_sd.pt"] + [os.path.basename(f) for f in
                                                                               glob.glob(os.path.join(out_dir, "z0_ddpm*.pt"))]
    for f in stale:                             # an earlier DDPM in this directory (removed once the stage 1 has loaded)
        if os.path.exists(os.path.join(out_dir, f)):
            os.remove(os.path.join(out_dir, f))
    shutil.rmtree(os.path.join(out_dir, "state_scores"), ignore_errors=True)
    stage1.requires_grad_(False)
    z0, n_list = encode_pool(stage1, tr, reps=reps, device=dev)
    z0_mu, z0_sd = z0.mean(0), z0.std(0).clamp_min(1e-6)
    ddpm = Z0DDPM(int(cfg["z0_dim"])).to(dev)
    zv, nv = encode_pool(stage1, va, reps=reps, device=dev)
    print(f"[z0-ddpm] {cfg['dataset']}: pool {z0.shape[0]} ({reps} per training graph), val pool {zv.shape[0]}", flush=True)
    t0 = time.time()
    losses, vlog = train_z0_ddpm(ddpm, (z0 - z0_mu) / z0_sd, torch.as_tensor(n_list), epochs, batch_size=batch_size, lr=lr,
                                 out_dir=out_dir, val_pool=(zv - z0_mu) / z0_sd, val_n=torch.as_tensor(nv),
                                 snapshots=snapshots, device=dev)
    atomic_save(ddpm.state_dict(), os.path.join(out_dir, "z0_ddpm.pt"))
    atomic_save(z0_mu, os.path.join(out_dir, "z0_mu.pt"))
    atomic_save(z0_sd, os.path.join(out_dir, "z0_sd.pt"))
    meta = dict(stage1=s1, stage1_ckpt=os.path.basename(s1), stage1_sha1=sha1, epochs=epochs, lr=lr, batch_size=batch_size,
                z0_reps=reps, seed=seed, minutes=(time.time() - t0) / 60, loss=losses, val=vlog)
    write_json(meta, os.path.join(out_dir, "ddpm_z0_meta.json"))     # written last: a DDPM without it is incomplete
    print(f"[z0-ddpm] done in {meta['minutes']:.1f} min, final eps-loss {losses[-1]:.5f} -> {out_dir}", flush=True)
    return out_dir
