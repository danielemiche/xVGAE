"""Train stage 1 with the standard recipe (or a paper row's config), then the validity gate.

    python scripts/train_stage1.py --data data --name mmsbm_medium --decoder mlp --run-dir runs/mmsbm_mlp
    python scripts/train_stage1.py --config configs/mmsbm_mlp.json --data data --run-dir runs/mmsbm_mlp
    python scripts/train_stage1.py --resume runs/mmsbm_mlp
Exit code: 0 valid, 10 collapse, 11 undertrained, 12 wall-time, 13 no trajectory.
"""
import argparse
import json
import os
import sys

from xvgae import set_threads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", "--dataset", dest="name", help="corpus name")
    ap.add_argument("--data", help="data root (default $XVGAE_DATA_DIR, else ./data)")
    ap.add_argument("--decoder", choices=["sab", "mlp"], help="default sab")
    ap.add_argument("--config", help="a paper row's config (configs/<row>.json)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run-dir")
    ap.add_argument("--updates", type=int, help="gradient updates (default 75k)")
    ap.add_argument("--max-walltime-h", type=float)
    ap.add_argument("--free-bits", type=float, help="KL floor of the node latents, nats per node and dimension")
    ap.add_argument("--beta", type=float, help="final weight of the node-latent KL (default 1)")
    ap.add_argument("--beta0", type=float, help="weight of the global-latent KL (default 0.1)")
    ap.add_argument("--snapshots", type=int, default=20, help="model_ep<E>.pt snapshots (0 = none)")
    ap.add_argument("--smoke", action="store_true", help="2 epochs: an installation check")
    ap.add_argument("--resume", metavar="RUN_DIR")
    ap.add_argument("--device")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    set_threads(a.threads)
    from xvgae import train as T
    from xvgae.data import set_data_root
    set_data_root(a.data)
    if a.resume:
        given = [f for f in ("name", "decoder", "config", "seed", "run_dir", "updates", "free_bits", "beta", "beta0")
                 if getattr(a, f) is not None]
        if given:
            ap.error(f"--resume continues the run as configured; not with {', '.join('--' + f.replace('_', '-') for f in given)}")
        return T.EXIT[T.resume(a.resume, device=a.device, max_walltime_h=a.max_walltime_h)[1]["kind"]]
    cfg = {k: v for k, v in json.load(open(a.config)).items() if not k.startswith("_")} if a.config else None
    name = a.name or (cfg or {}).get("dataset")
    if not name:
        ap.error("--name (or --config) is required")
    if cfg:
        fixed = [f for f in ("decoder", "updates", "beta0") if getattr(a, f) is not None]
        if fixed:
            ap.error(f"the config fixes {', '.join('--' + f for f in fixed)}: change them in a copy of the config")
    if a.run_dir and not a.smoke and os.path.exists(os.path.join(a.run_dir, "config.json")):
        ap.error(f"{a.run_dir} already holds a run: --resume {a.run_dir}, or another --run-dir")
    _, v = T.train_standard(name, decoder=(cfg or {}).get("decoder", a.decoder or "sab"), seed=a.seed, run_dir=a.run_dir, config=cfg,
                            updates=a.updates, max_walltime_h=a.max_walltime_h, device=a.device, smoke=a.smoke,
                            snapshots=a.snapshots, free_bits=a.free_bits, beta=a.beta, beta0=a.beta0)
    return 0 if a.smoke else T.EXIT[v["kind"]]


if __name__ == "__main__":
    sys.exit(main())
