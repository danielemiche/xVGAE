"""Stage 1 training: the standard recipe, the training loop and the validity gate.

Per epoch: KL weights and the cosine learning rate from the schedule; per batch random eigenvector sign flips, the
negative ELBO, gradient clip 1.0, AdamW. Every val_freq epochs: validation loss and edge ROC, model_last.pt,
model_best_warmup.pt (lowest validation loss during the KL warm-up) and snapshots model_ep<E>.pt; every val_mmd_freq
epochs: model_best_gen.pt (lowest validation MMD of encoded z0 + prior z, geometric mean of the ratios to the selection
floor). resume.pt every 15 minutes; VALIDITY.json at the end.
"""
import json
import math
import os
import time

import networkx as nx
import numpy as np
import torch

from xvgae import atomic_save, pick_device
from xvgae.layers import tril_indices
from xvgae.model import ARCH, build_model

RECIPE = dict(seed=0, batch_size=8, epochs=None, lr=5e-4, lr_min=2.5e-5, lr_schedule="cosine", weight_decay=1e-5,
              anneal=False, warmup_epochs=0, anneal_epochs=0, beta=1.0, beta0=0.1, free_bits_z=None, val_freq=5,
              val_mmd=True, val_mmd_freq=25, val_mmd_metrics=["degree", "clustering", "spectral", "wl"])
RUNTIME = {"dataset", "_data_root", "max_walltime_sec", "snapshot_every"}
UPDATES, MAX_HOURS, KL_WARMUP, KL_WARMUP_MIN_UPDATES = 75_000, 10.0, 0.05, 3750
EXIT = {"valid": 0, "collapse": 10, "undertrained": 11, "walltime": 12, "missing": 13}
NEXT = {"collapse": "retrain once with --free-bits 0.15 and twice the --updates",
        "undertrained": "retrain with more --updates", "walltime": "continue with --resume <run dir>",
        "missing": "check the log"}


def standard_config(dataset, decoder="sab", seed=0, beta0=0.1):
    """The standard recipe for one corpus and decoder (finalise_budget sets the epochs and the KL warm-up)."""
    return {**ARCH, **RECIPE, "dataset": dataset, "decoder": decoder, "seed": int(seed), "beta0": float(beta0)}


def finalise_budget(cfg, updates_per_epoch, updates=UPDATES, epochs=None, kl_warmup=KL_WARMUP,
                    min_updates=KL_WARMUP_MIN_UPDATES):
    """Epochs from the update budget; KL warm-up over kl_warmup of them, at least min_updates updates (at most half the
    epochs): a quarter at 1e-4, then a half-cosine ramp."""
    full = int(epochs) if epochs else math.ceil(updates / updates_per_epoch)
    if kl_warmup > 0 and full >= 4:
        span = min(max(2, round(kl_warmup * full)), full - 1)
        if min_updates:
            span = max(span, min(min_updates // max(1, int(updates_per_epoch)), full // 2))
        flat = max(1, round(0.25 * span))
        cfg.update(anneal=True, warmup_epochs=flat, anneal_epochs=max(1, span - flat))
    cfg["epochs"] = full
    return cfg


def kl_weights(epoch, cfg):
    """(beta, beta0): constant without a warm-up; else 1e-4 for warmup_epochs, then a half-cosine to beta over
    anneal_epochs, with beta0 scaled by beta."""
    if not cfg["anneal"]:
        return float(cfg["beta"]), float(cfg["beta0"])
    t = epoch - cfg["warmup_epochs"]
    if t < 0:
        beta = 1e-4
    elif t >= cfg["anneal_epochs"]:
        beta = cfg["beta"]
    else:
        beta = cfg["beta"] - (cfg["beta"] - 1e-4) * 0.5 * (1 + np.cos(np.pi * t / cfg["anneal_epochs"]))
    return beta, float(cfg["beta0"]) * float(beta)


def lr_factor(epoch, cfg):
    """1 (constant) or a cosine from 1 to lr_min / lr over the epochs."""
    if cfg["lr_schedule"] == "constant":
        return 1.0
    f = cfg["lr_min"] / cfg["lr"]
    return f + 0.5 * (1.0 - f) * (1.0 + np.cos(np.pi * min(1.0, epoch / max(1, cfg["epochs"]))))


def sign_augment(batch):
    """Flip the sign of each eigenvector per graph with probability 1/2 (x and eigvecs alike)."""
    B = int(batch.batch.max().item()) + 1
    s = (torch.randint(0, 2, (B, batch.x.shape[-1]), device=batch.x.device).float() * 2.0 - 1.0)[batch.batch]
    batch.x, batch.eigvecs = batch.x * s, batch.eigvecs * s
    return batch


def edge_scores(out, batch):
    """(true edges, edge probabilities) over the valid node pairs of a batch."""
    from torch_geometric.utils import to_dense_adj
    mask = out["mask"]
    N = mask.shape[1]
    adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=N)
    adj = torch.clamp(adj + adj.transpose(1, 2), 0, 1)
    ti, tj = tril_indices(N, mask.device)
    pm = mask[:, ti] & mask[:, tj]
    return adj[:, ti, tj][pm].cpu(), torch.sigmoid(out["logits_vec"])[pm].cpu()


@torch.no_grad()
def validate(model, loader, beta, beta0, cfg):
    """Mean validation loss, recon, kl_z, kl_z0, and the edge ROC AUC."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    acc, ys, ps = {}, [], []
    for batch in loader:
        batch = batch.to(model.device)
        out = model(batch)
        for k, v in model.compute_loss(out, batch, beta=beta, beta0=beta0, free_bits_z=cfg["free_bits_z"])[1].items():
            acc[k] = acc.get(k, 0.0) + v
        y, p = edge_scores(out, batch)
        ys.append(y)
        ps.append(p)
    vs = {k: v / len(loader) for k, v in acc.items()}
    y, p = torch.cat(ys).numpy() > 0.5, torch.cat(ps).numpy()
    vs["roc_auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < y.size and np.isfinite(p).all() else float("nan")
    return vs


def selection_floor(dataset):
    """{statistic: floor} of the validation MMD: the paper's for a paper corpus, else <data>/<name>/floor_selection.json."""
    from xvgae.data import REFERENCE, data_root
    per = json.load(open(REFERENCE))["selection_floor"].get(dataset)
    p = os.path.join(data_root(), str(dataset), "floor_selection.json")
    if per is None and os.path.exists(p):
        per = json.load(open(p))["per_seed"]
    return {m: float(np.mean(v)) for m, v in (per or {}).items()}


def gen_score(mmd, floor):
    """Geometric mean of MMD^2 / floor (of the raw MMD^2 without a floor)."""
    v = [mmd[m] / floor[m] if floor.get(m) else mmd[m] for m in mmd]
    return float(np.exp(np.mean(np.log(np.maximum(v, 1e-300)))))


def train(model, train_loader, val_loader, cfg, run_dir):
    """Train stage 1 in run_dir (config.json, checkpoints, trajectory_local.json, run_meta.json). Returns the model."""
    from xvgae.metrics import pyg_to_nx, val_stats, val_mmd
    os.makedirs(run_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    E, vf = int(cfg["epochs"]), int(cfg["val_freq"])
    anneal_end = cfg["warmup_epochs"] + cfg["anneal_epochs"] if cfg["anneal"] else 0
    floor = selection_floor(cfg["dataset"])
    ckpt = lambda name: atomic_save(model.state_dict(), os.path.join(run_dir, name))  # noqa: E731
    hist = {k: [] for k in ("train/loss", "train/recon", "train/kl_z", "train/kl_z0", "val/loss", "val/recon", "val/kl_z",
                            "val/kl_z0", "val/roc_auc")}
    best = dict(warm=float("inf"), gen=float("inf"), gen_ep=None, last_snap=0)
    start, t0, t_resume, walltime = 0, time.time(), time.time(), False
    with torch.random.fork_rng(devices=[]):                     # the validation reference set, without an RNG draw
        val_ref = val_stats(pyg_to_nx(val_loader), cfg["val_mmd_metrics"]) if cfg["val_mmd"] else None
    resume_path = os.path.join(run_dir, "resume.pt")
    if os.path.exists(resume_path):
        R = torch.load(resume_path, map_location=model.device, weights_only=False)
        model.load_state_dict(R["model"])
        opt.load_state_dict(R["optimizer"])
        hist, best, start = R["hist"], R["best"], R["epoch"] + 1
        torch.set_rng_state(R["rng"]["torch"])
        np.random.set_state(R["rng"]["numpy"])
        if R["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(R["rng"]["cuda"])
        print(f"[train] resumed at epoch {start}/{E}", flush=True)

    def save_resume(epoch):
        rng = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        atomic_save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(), hist=hist, best=best, rng=rng),
                    resume_path)

    epoch = start - 1
    for epoch in range(start, E):
        t_ep = time.time()
        beta, beta0 = kl_weights(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * lr_factor(epoch, cfg)
        model.train()
        acc = {}
        for batch in train_loader:
            batch = sign_augment(batch.to(model.device))
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss, st = model.compute_loss(out, batch, beta=beta, beta0=beta0, free_bits_z=cfg["free_bits_z"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}: {st}")
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch}")
            opt.step()
            for k in ("loss", "recon", "kl_z", "kl_z0"):
                acc[k] = acc.get(k, 0.0) + st[k]
        for k, v in acc.items():
            hist[f"train/{k}"].append(v / len(train_loader))
        print(f"[train] epoch {epoch}/{E} loss {hist['train/loss'][-1]:.3f} recon {hist['train/recon'][-1]:.4f} kl_z "
              f"{hist['train/kl_z'][-1]:.4f} kl_z0 {hist['train/kl_z0'][-1]:.4f} beta {beta:.4g} ({time.time() - t_ep:.1f} s)",
              flush=True)
        vs = None
        if epoch % vf == 0 or epoch == E - 1:
            vs = validate(model, val_loader, beta, beta0, cfg)
            print(f"  [val] loss={vs['loss']:.4f} kl_z={vs['kl_z']:.4f} kl_z0={vs['kl_z0']:.4f} roc_auc={vs['roc_auc']:.4f}",
                  flush=True)
            if epoch < anneal_end and vs["loss"] < best["warm"]:
                best["warm"] = vs["loss"]
                ckpt("model_best_warmup.pt")
            ckpt("model_last.pt")
            snap = int(cfg.get("snapshot_every") or 0)
            if snap > 0 and (epoch + 1) - best["last_snap"] >= snap:
                ckpt(f"model_ep{epoch + 1}.pt")
                best["last_snap"] = epoch + 1
            if cfg["val_mmd"] and (epoch % int(cfg["val_mmd_freq"]) == 0 or epoch == E - 1):
                mmd = val_mmd(val_ref, [nx.from_numpy_array(a) for a in model.encode_and_generate(val_loader)])
                score = gen_score(mmd, floor)
                print("  [val MMD] " + "  ".join(f"{k}={v:.4f}" for k, v in mmd.items()) + f"  ratio geomean {score:.3f}",
                      flush=True)
                if score < best["gen"]:
                    best["gen"], best["gen_ep"] = score, epoch
                    ckpt("model_best_gen.pt")
        for k in ("loss", "recon", "kl_z", "kl_z0", "roc_auc"):
            hist[f"val/{k}"].append(vs[k] if vs is not None else None)
        if vs is not None:
            json.dump(hist, open(os.path.join(run_dir, "trajectory_local.json"), "w"))
        if epoch < E - 1 and time.time() - t_resume >= 900:
            save_resume(epoch)
            t_resume = time.time()
        if cfg.get("max_walltime_sec") and time.time() - t0 >= cfg["max_walltime_sec"]:
            walltime = epoch < E - 1
            if walltime:
                save_resume(epoch)
            print(f"[train] wall-time budget reached after epoch {epoch} (continue with --resume)", flush=True)
            break
    ckpt("model_last.pt")
    if not walltime and os.path.exists(resume_path):
        os.remove(resume_path)
    json.dump(dict(last_epoch=int(epoch), total_epochs=E, walltime=walltime, best_gen_epoch=best["gen_ep"]),
              open(os.path.join(run_dir, "run_meta.json"), "w"), indent=1)
    json.dump(hist, open(os.path.join(run_dir, "trajectory_local.json"), "w"))
    print(f"[train] done: {os.path.abspath(run_dir)} (best gen epoch {best['gen_ep']})", flush=True)
    return model


def check_validity(run_dir, quiet=False):
    """VALIDITY.json: collapse (kl_z < 0.05 or validation ROC < 0.65), undertrained (validation loss still falling > 1 %
    over the last 10 % of the epochs), walltime (stopped early), missing (no trajectory)."""
    tp, mp = os.path.join(run_dir, "trajectory_local.json"), os.path.join(run_dir, "run_meta.json")
    last = lambda v: next((x for x in reversed(v or []) if x is not None), None)  # noqa: E731
    reasons, warn, nums, kind = [], [], {}, None
    if not os.path.exists(tp):
        reasons, kind = ["no trajectory_local.json"], "missing"
    else:
        t = json.load(open(tp))
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        E = int(m.get("total_epochs") or len(t["train/loss"]))
        kz = last(t["val/kl_z"]) if last(t["val/kl_z"]) is not None else last(t["train/kl_z"])
        roc = last(t["val/roc_auc"])
        w = [x for e, x in enumerate(t["val/loss"]) if x is not None and e >= 0.9 * (E - 1)]
        impr = (w[0] - w[-1]) / abs(w[0]) if len(w) >= 2 and w[0] else None
        nums = dict(kl_z=kz, val_roc=roc, val_loss_improvement_last10pct=impr, last_epoch=m.get("last_epoch"), epochs=E)
        if not m or m.get("walltime") or os.path.exists(os.path.join(run_dir, "resume.pt")) or m["last_epoch"] < E - 1:
            reasons.append("walltime: the run stopped before its last epoch")
            kind = "walltime"
        if (kz is not None and kz < 0.05) or (roc is not None and roc < 0.65):
            reasons.append(f"collapse: kl_z {kz} (< 0.05?), val ROC {roc} (< 0.65?)")
            kind = kind or "collapse"
        if impr is not None and impr > 0.01:
            reasons.append(f"undertrained: val loss still -{100 * impr:.1f}% over the last 10% of the epochs")
            kind = kind or "undertrained"
        if E < 20:
            warn.append(f"only {E} epochs: the verdict says nothing about the recipe")
    r = dict(valid=not reasons, kind=kind or "valid", reasons=reasons, numbers=nums, warnings=warn)
    json.dump(r, open(os.path.join(run_dir, "VALIDITY.json"), "w"), indent=1)
    if not quiet:
        print(f"[validity] {run_dir}: " + ("VALID" if r["valid"] else f"NOT A MODEL ({'; '.join(reasons)}): "
                                           f"{NEXT[r['kind']]}"), flush=True)
    return r


def train_standard(dataset, decoder="sab", seed=None, run_dir=None, config=None, updates=None, max_walltime_h=None,
                   device=None, smoke=False, snapshots=20, free_bits=None, beta=None, beta0=None):
    """Train one stage 1 with the standard recipe (or a paper row's config) and run the validity gate.
    Returns (run_dir, verdict)."""
    from xvgae.data import get_dataloaders, data_root
    dev = pick_device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = True
    elif not smoke:
        print("[train] no GPU: training on the CPU (slow at the full budget; see --updates in docs/TRAINING.md)", flush=True)
    seed = int((config or {}).get("seed", 0) if seed is None else seed)
    cfg = standard_config(dataset, decoder, seed, beta0=0.1 if beta0 is None else beta0)
    # a config's batch size builds the loaders (and so the updates per epoch of the budget), as resume() does
    tr, va, _ = get_dataloaders(dataset, int((config or {}).get("batch_size", cfg["batch_size"])))
    cfg["_data_root"] = os.path.abspath(data_root())
    finalise_budget(cfg, len(tr), updates=updates or UPDATES)
    if config:
        unknown = set(config) - set(ARCH) - set(RECIPE) - RUNTIME
        if unknown:
            raise ValueError(f"unknown config keys {sorted(unknown)}")
        cfg.update(config, seed=seed)
    vf = cfg["val_freq"]
    cfg["snapshot_every"] = max(vf, int(round(cfg["epochs"] / snapshots / vf)) * vf) if snapshots else 0
    cfg["max_walltime_sec"] = 3600 * (MAX_HOURS if max_walltime_h is None else max_walltime_h)
    if free_bits is not None:
        cfg["free_bits_z"] = float(free_bits) or None
    if beta is not None:
        cfg["beta"] = float(beta)
    if smoke:
        cfg.update(epochs=2, val_mmd=False)
    run_dir = run_dir or os.path.join("runs", f"{dataset}_{decoder}_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}")
    warm = cfg["warmup_epochs"] + cfg["anneal_epochs"] if cfg["anneal"] else 0
    print(f"[train] {dataset} decoder {cfg['decoder']} seed {seed}: {cfg['epochs']} epochs "
          f"({cfg['epochs'] * len(tr)} updates), "
          f"KL warm-up {warm} epochs, beta0 {cfg['beta0']}, KL floor {cfg['free_bits_z']}, device {dev} -> {run_dir}",
          flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train(build_model(cfg, tr, dev), tr, va, cfg, run_dir)
    return run_dir, check_validity(run_dir, quiet=smoke)


def resume(run_dir, device=None, max_walltime_h=None):
    """Continue a stopped run from its resume.pt. Returns (run_dir, verdict)."""
    from xvgae.data import get_dataloaders
    cfg = {**ARCH, **RECIPE, **json.load(open(os.path.join(run_dir, "config.json")))}
    if cfg.get("_data_root") and "XVGAE_DATA_DIR" not in os.environ:
        os.environ["XVGAE_DATA_DIR"] = cfg["_data_root"]
    if not os.path.exists(os.path.join(run_dir, "resume.pt")):
        raise FileNotFoundError(f"{run_dir}/resume.pt not found: the run finished or never saved one")
    if max_walltime_h is not None:
        cfg["max_walltime_sec"] = 3600 * max_walltime_h
    dev = pick_device(device)
    tr, va, _ = get_dataloaders(cfg["dataset"], cfg["batch_size"])
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    train(build_model(cfg, tr, dev), tr, va, cfg, run_dir)
    return run_dir, check_validity(run_dir)
