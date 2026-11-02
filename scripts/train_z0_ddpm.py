"""Train the n-conditioned z0 DDPM (stage 2) on a trained stage 1.

    python scripts/train_z0_ddpm.py --run runs/mmsbm_mlp --out-dir runs/mmsbm_mlp/z0_ddpm
The stage-1 checkpoint is the validation pick of scripts/select_stage1.py (--ckpt names another).
Paper recipe: 20 posterior z0 draws per training graph, 2000 epochs, lr 2e-4, batch 64; snapshots at epochs 1000 and
2000 and the best held-out state (scripts/sample.py picks one of them on validation).
"""
import argparse

from xvgae import set_threads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="stage-1 run directory")
    ap.add_argument("--ckpt", default="auto", help="stage-1 checkpoint (default: the select_stage1.py pick)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", help="data root (default: the run's)")
    ap.add_argument("--device")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    set_threads(a.threads)
    from xvgae.data import set_data_root
    from xvgae.diffusion import fit
    set_data_root(a.data)
    fit(a.run, a.ckpt, a.out_dir, epochs=a.epochs, seed=a.seed, device=a.device)


if __name__ == "__main__":
    main()
