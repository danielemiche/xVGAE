"""Draw graphs from a trained xVGAE (stage 1 + the z0 DDPM): one .npz per seed, |split| graphs each.

    python scripts/sample.py --run runs/mmsbm_mlp --ddpm-dir runs/mmsbm_mlp/z0_ddpm \\
                             --seeds 100,101,102 --out-dir samples/mmsbm_mlp
The stage 1 is the one the DDPM was trained on (its ddpm_z0_meta.json). The DDPM state (the epoch snapshots and the best
held-out state) is picked on validation the first time: 10 validation draws per state (seeds 0-9, z0 from the DDPM, n from
the training sizes), each scored against the validation split and averaged by xvgae.metrics.Selector (EMD kernel, geometric
mean of the five MMD^2, states within 10 % tied, the C2ST moves the pick only beyond noise). The pick is kept in
<ddpm-dir>/state_pick.json.
--no-ddpm draws z0 from the stage-1 prior p(z0 | n) instead (write it to its own --out-dir: score.py scores a directory).
"""
import argparse
import json
import os
import re
import time

import numpy as np

from xvgae import file_id, file_sha1, set_threads, write_json


def read_json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def pick_state(model, va, hist, ddpm_dir, z0_dim, sel, device, stage1, val_draws=10):
    """The DDPM state with the best validation score (written to state_pick.json). Each state's score is kept in
    <ddpm-dir>/state_scores/, so a stopped pick continues where it was."""
    from xvgae.diffusion import load_z0_ddpm
    from xvgae.model import draw
    seeds = list(range(val_draws))
    eps = sorted((int(m.group(1)), f) for f in os.listdir(ddpm_dir) if (m := re.fullmatch(r"z0_ddpm_ep(\d+)\.pt", f)))
    cache = os.path.join(ddpm_dir, "state_scores")
    os.makedirs(cache, exist_ok=True)
    res = {}
    for st in [f for _, f in eps] + ["z0_ddpm_bestval.pt"]:
        if not os.path.exists(os.path.join(ddpm_dir, st)):
            continue
        cp, fid = os.path.join(cache, st.replace(".pt", ".json")), file_id(os.path.join(ddpm_dir, st))
        prev = read_json(cp)
        if prev.get("file") == fid and prev.get("seeds") == seeds and prev.get("rule") == sel.rule and prev.get("stage1") == stage1:
            r = prev["score"]
        else:
            dd = load_z0_ddpm(ddpm_dir, z0_dim, device=device, state=st)
            r = sel.score([draw(model, va, hist, s, ddpm=dd, device=device) for s in seeds])
            write_json(dict(file=fid, seeds=seeds, rule=sel.rule, stage1=stage1, score=r), cp)
        res[st] = r
        print(f"[pick] {st}: val MMD^2 geomean {r['geomean']:.5g}  C2ST {r['c2st_acc']:.3f} ± {r['c2st_se']:.3f}", flush=True)
    pick, tied, d = sel.pick(res)
    write_json(dict(pick=pick, tied=tied, c2st_d=d, mode="fully (z0 from the DDPM, n from the training sizes)", seeds=seeds,
                    stage1=stage1, dataset=sel.dataset, rule=sel.rule, scores=res), os.path.join(ddpm_dir, "state_pick.json"))
    print(f"[pick] -> {pick} (tied: {tied})", flush=True)
    return pick


def stage1_of(ddpm_dir, run, ckpt):
    """The path of the stage-1 checkpoint a DDPM was trained on: the name in its ddpm_z0_meta.json, in --run, checked against
    the recorded sha1. An explicit --ckpt must be that file."""
    mp = os.path.join(ddpm_dir, "ddpm_z0_meta.json")
    meta = read_json(mp)
    used = meta.get("stage1_ckpt") or (os.path.basename(meta["stage1"]) if meta.get("stage1") else None)
    if used in (None, "auto"):
        if ckpt == "auto":
            raise SystemExit(f"{mp}: no record of the DDPM's stage 1; pass --ckpt, or retrain the DDPM (train_z0_ddpm.py)")
        print(f"[sample] {mp}: no record of the DDPM's stage 1, --ckpt {ckpt} used unchecked", flush=True)
        return ckpt if os.path.dirname(ckpt) else os.path.join(run, ckpt)
    path = os.path.join(run, used) if ckpt == "auto" else (ckpt if os.path.dirname(ckpt) else os.path.join(run, ckpt))
    if os.path.basename(path) != used:
        raise SystemExit(f"--ckpt {ckpt}, but the DDPM in {ddpm_dir} was trained on {used}")
    if meta.get("stage1_sha1") and file_sha1(path) != meta["stage1_sha1"]:
        raise SystemExit(f"{path} is not the file the DDPM in {ddpm_dir} was trained on (sha1 differs from {mp})")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="auto",
                    help="stage-1 checkpoint (default: the one the DDPM was trained on; with --no-ddpm the select_stage1.py pick)")
    ap.add_argument("--ddpm-dir")
    ap.add_argument("--ddpm-state", help="a state file in --ddpm-dir (default: the validation pick)")
    ap.add_argument("--val-draws", type=int, default=10, help="validation draws per DDPM state for the pick")
    ap.add_argument("--no-ddpm", action="store_true")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--seeds", default="100,101,102")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data", help="data root (default: the run's)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    set_threads(a.threads)
    from xvgae.data import set_data_root, train_sizes
    from xvgae.diffusion import load_z0_ddpm
    from xvgae.metrics import Selector, write_npz
    from xvgae.model import draw, enforce_fp32, load_stage1, resolve_ckpt
    if not (a.no_ddpm or a.ddpm_dir):
        ap.error("--ddpm-dir (or --no-ddpm)")
    set_data_root(a.data)
    enforce_fp32()
    if a.no_ddpm:
        ckpt = resolve_ckpt(a.run, a.ckpt)
        ckpt = ckpt if os.path.dirname(ckpt) else os.path.join(a.run, ckpt)
    else:
        ckpt = stage1_of(a.ddpm_dir, a.run, a.ckpt)
    m, cfg, tr, va, te = load_stage1(a.run, ckpt, device=a.device)
    hist = train_sizes(tr)
    dd, state = None, a.ddpm_state
    if not a.no_ddpm and not state:
        s1 = dict(ckpt=os.path.basename(ckpt), sha1=file_sha1(ckpt))
        sel = Selector(cfg["dataset"])
        pk = os.path.join(a.ddpm_dir, "state_pick.json")
        prev = read_json(pk)
        if prev.get("stage1") == s1 and prev.get("rule") == sel.rule and prev.get("seeds") == list(range(a.val_draws)):
            state = prev["pick"]
        else:
            if prev:
                print(f"[pick] {pk} was made for another stage 1, rule or draw count: picking again", flush=True)
            state = pick_state(m, va, hist, a.ddpm_dir, cfg["z0_dim"], sel, a.device, s1, a.val_draws)
    if not a.no_ddpm:
        dd = load_z0_ddpm(a.ddpm_dir, cfg["z0_dim"], device=a.device, state=state)
    os.makedirs(a.out_dir, exist_ok=True)
    meta_p = os.path.join(a.out_dir, "meta.json")
    meta = read_json(meta_p)
    for s in (int(x) for x in a.seeds.split(",")):
        t0 = time.time()
        gs = draw(m, va if a.split == "val" else te, hist, s, ddpm=dd, device=a.device)
        p = os.path.join(a.out_dir, f"{'nodiff' if a.no_ddpm else 'xvgae'}_s{s}.npz")
        write_npz(gs, p)
        r = meta[os.path.basename(p)] = dict(dataset=cfg["dataset"], split=a.split, seed=s, graphs=len(gs),
                                             seconds=round(time.time() - t0, 1), stage1=os.path.abspath(ckpt),
                                             z0_ddpm=os.path.join(os.path.abspath(a.ddpm_dir), state) if dd else None,
                                             mean_n=float(np.mean([g.number_of_nodes() for g in gs])),
                                             mean_m=float(np.mean([g.number_of_edges() for g in gs])))
        write_json(meta, meta_p)
        print(f"[sample] {p}: {len(gs)} graphs, mean n {r['mean_n']:.1f}, mean m {r['mean_m']:.1f} ({r['seconds']} s)",
              flush=True)


if __name__ == "__main__":
    main()
