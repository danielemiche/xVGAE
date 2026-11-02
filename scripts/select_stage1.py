"""Pick the stage-1 checkpoint on validation (written to <run>/stage1_pick.json).

    python scripts/select_stage1.py --run runs/mmsbm_mlp
Candidates: every model_*.pt of the run (the epoch snapshots and the best / last states). Each draws 3 validation sets
(seeds 0-2) with z0 from the encoder (one draw of q(z0 | G) per validation graph, at that graph's n) and the node latents from
p(z | z0); each set is scored against the validation split and the scores are averaged over the sets by
xvgae.metrics.Selector (EMD kernel, geometric mean of the five MMD^2, candidates within 10 % tied, the C2ST moves the pick
only beyond noise). train_z0_ddpm.py and sample.py then use the pick (their --ckpt default). The run must have passed the
validity check of train_stage1.py (VALIDITY.json), and so must each candidate on the validation split (edge ROC AUC >= 0.65,
kl_z >= 0.05 nats per node and dimension); the others are left out. --skip-warmup leaves out the checkpoints written inside
the KL warm-up. Each candidate's score is kept in <run>/stage1_scores/, so a stopped run continues where it was.
"""
import argparse
import glob
import json
import os
import re

from xvgae import file_id, set_threads, write_json


def draw_encoded(model, loader, seed, device):
    """One validation draw: z0 ~ q(z0 | G) per graph, z ~ p(z | z0), n = the graph's own size."""
    import torch
    from xvgae.layers import reparameterize
    from xvgae.model import edges_from_logits
    torch.manual_seed(seed)
    graphs = []
    with torch.no_grad():
        for b in loader:
            out = model(b.to(device))
            ns = [int(x) for x in out["mask"].sum(1).tolist()]
            o = model.sample_logits(ns, z0=reparameterize(out["mu0"], out["lv0"]))
            graphs += [edges_from_logits(o["logits_vec"][i], n) for i, n in enumerate(ns)]
    return graphs


def candidates(run, cfg, skip_warmup):
    """The checkpoint names to screen, and those --skip-warmup leaves out."""
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run, "model_*.pt")))
    if not skip_warmup:
        return names, []
    end = cfg.get("warmup_epochs", 0) + cfg.get("anneal_epochs", 0)      # epochs (0-based) < end are below the full KL weight
    meta = os.path.join(run, "run_meta.json")
    gen_ep = json.load(open(meta)).get("best_gen_epoch") if os.path.exists(meta) else None
    warm = [f for f in names if f == "model_best_warmup.pt" or (f == "model_best_gen.pt" and (gen_ep is None or gen_ep < end))
            or ((m := re.fullmatch(r"model_ep(\d+)\.pt", f)) and int(m.group(1)) <= end)]
    return [f for f in names if f not in warm], warm


def gate(model, loader, cfg):
    """The validity check of one checkpoint on the validation split (the thresholds of xvgae.train.check_validity)."""
    import math
    import torch
    from xvgae.train import validate
    torch.manual_seed(0)
    with torch.no_grad():
        v = validate(model, loader, cfg["beta"], cfg["beta0"], {**cfg, "free_bits_z": cfg.get("free_bits_z")})
    roc, kz = float(v["roc_auc"]), float(v["kl_z"])
    bad = [why for ok, why in ((math.isfinite(roc) and roc >= 0.65, f"val ROC {roc:.3f} < 0.65"),
                               (kz >= 0.05, f"kl_z {kz:.4f} < 0.05")) if not ok]
    return dict(roc_auc=roc, kl_z=kz, valid=not bad, reasons=bad)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="stage-1 run directory")
    ap.add_argument("--skip-warmup", action="store_true", help="no checkpoint from inside the KL warm-up")
    ap.add_argument("--draws", type=int, default=3, help="validation sets per candidate (seeds 0..draws-1)")
    ap.add_argument("--force", action="store_true", help="screen a run that did not pass the validity check")
    ap.add_argument("--data", help="data root (default: the run's)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    set_threads(a.threads)
    from xvgae.data import set_data_root
    from xvgae.metrics import Selector
    from xvgae.model import enforce_fp32, load_stage1
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    vp = os.path.join(a.run, "VALIDITY.json")
    validity = json.load(open(vp)) if os.path.exists(vp) else {"valid": False, "reasons": ["no VALIDITY.json"]}
    if not validity.get("valid") and not a.force:
        raise SystemExit(f"{a.run}: not a valid stage 1 ({validity.get('kind') or validity.get('reasons')}); retrain it, "
                         f"or --force to screen it anyway")
    set_data_root(a.data)
    if "XVGAE_DATA_DIR" not in os.environ and os.path.isdir(cfg.get("_data_root") or ""):
        set_data_root(cfg["_data_root"])
    enforce_fp32()
    names, warm = candidates(a.run, cfg, a.skip_warmup)
    if warm:
        print(f"[select] --skip-warmup leaves out {warm}", flush=True)
    sel, res = Selector(cfg["dataset"]), {}
    cache = os.path.join(a.run, "stage1_scores")
    os.makedirs(cache, exist_ok=True)
    seeds = list(range(a.draws))
    gated = {}
    for c in names:
        cp, fid = os.path.join(cache, c.replace(".pt", ".json")), file_id(os.path.join(a.run, c))
        try:
            prev = json.load(open(cp))
        except (OSError, ValueError):
            prev = {}
        if prev.get("file") == fid and prev.get("seeds") == seeds and prev.get("rule") == sel.rule:
            g, r = prev["gate"], prev["score"]
        else:
            m, mcfg, _, va, _ = load_stage1(a.run, c, device=a.device)
            g = gate(m, va, mcfg)
            r = sel.score([draw_encoded(m, va, s, a.device) for s in seeds]) if g["valid"] else None
            write_json(dict(file=fid, seeds=seeds, rule=sel.rule, gate=g, score=r), cp)
        if not g["valid"]:
            gated[c] = g
            print(f"[select] {c}: left out, {'; '.join(g['reasons'])}", flush=True)
            continue
        res[c] = r
        print(f"[select] {c}: val MMD^2 geomean {r['geomean']:.5g}  C2ST {r['c2st_acc']:.3f} ± {r['c2st_se']:.3f}  "
              f"(ROC {g['roc_auc']:.3f}, kl_z {g['kl_z']:.3f})", flush=True)
    if not res:
        raise SystemExit(f"{a.run}: no checkpoint passes the validity check ({gated})")
    pick, tied, d = sel.pick(res)
    write_json(dict(pick=pick, tied=tied, c2st_d=d, mode="encz0gen", seeds=seeds, skip_warmup=a.skip_warmup, left_out=warm,
                    gate_out=gated, dataset=cfg["dataset"], data_root=os.environ.get("XVGAE_DATA_DIR"), rule=sel.rule,
                    scores=res), os.path.join(a.run, "stage1_pick.json"))
    print(f"[select] -> {pick} (tied: {tied})", flush=True)


if __name__ == "__main__":
    main()
