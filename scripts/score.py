"""Score sample files (one .npz per draw) and aggregate them into a table row.

    python scripts/score.py --name mmsbm_medium --data data --samples samples/mmsbm_mlp --out results/mmsbm_mlp.json
Per draw: MMD^2 of degree, clustering, spectral, orbit and WL against the test split (LCC) and the C2ST accuracy. The
default kernel is a Gaussian on the earth mover's distance with bandwidths fitted on the training graphs, so that no
statistic saturates; --kernel tv gives the Gaussian-TV kernels of the GraphRNN protocol. Ratio = MMD^2 / train-vs-test
floor; row = median over the draws [min, max]; Agg = geometric mean over the ranked statistics.
"""
import argparse
import glob
import json
import os
import re
import time

from xvgae import set_threads, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", "--dataset", dest="name", required=True, help="corpus name")
    ap.add_argument("--data", help="data root (default $XVGAE_DATA_DIR, else ./data)")
    ap.add_argument("--samples", required=True, help="comma-separated .npz files or directories")
    ap.add_argument("--kernel", choices=("emd", "tv"), default="emd")
    ap.add_argument("--sigma-clustering", choices=("fitted", "published"), default="fitted",
                    help="TV clustering bandwidth: median train TV distance, or 0.1")
    ap.add_argument("--saturation", choices=("off", "report", "exclude"), default="report",
                    help="1 - kbar per statistic on the test set (^ when >= 0.9); exclude also leaves those out of Agg")
    ap.add_argument("--measure-floor", action="store_true", help="measure the floor of a paper corpus")
    ap.add_argument("--out")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    set_threads(a.threads)
    from xvgae import metrics as M
    from xvgae.data import set_data_root
    set_data_root(a.data)
    files = [f for p in a.samples.split(",") if p
             for f in (sorted(glob.glob(os.path.join(p, "*.npz"))) if os.path.isdir(p) else [p])]
    if not files:
        raise SystemExit(f"--samples {a.samples}: no .npz file")
    models = sorted({m.group(1) for f in files if (m := re.fullmatch(r"(.+)_s\d+\.npz", os.path.basename(f)))})
    if len(models) > 1:     # sample.py names its draws <model>_s<seed>.npz
        raise SystemExit(f"--samples {a.samples}: draws of more than one model ({', '.join(models)}); one row per model, "
                         f"pass its files")
    recs = [json.load(open(mp)).get(os.path.basename(f)) for f in files
            if os.path.exists(mp := os.path.join(os.path.dirname(f), "meta.json"))]
    if recs and (None in recs or len({(r.get("stage1"), r.get("z0_ddpm"), r.get("split")) for r in recs}) > 1
                 or len({r.get("seed") for r in recs}) < len(recs)):
        raise SystemExit(f"--samples {a.samples}: the draws (meta.json) are not one model and split with distinct seeds")
    if recs and recs[0].get("split") != "test":
        raise SystemExit(f"--samples {a.samples}: draws of the {recs[0].get('split')} split; score.py scores against test")
    ref = M.paper_reference(a.name)
    paper_kernels = a.kernel == "tv" and a.sigma_clustering == "fitted"
    measure = a.measure_floor or ref is None or not paper_kernels
    orbit = bool(ref) or M.orbit_available()
    if not orbit:
        print("[score] ORCA not built: no orbit MMD (bash scripts/build_orca.sh)", flush=True)
    t0 = time.time()
    test, train = M.real_split(a.name, "test"), M.real_split(a.name, "train")
    if a.sigma_clustering == "published":
        sigma = M.PUBLISHED_SIGMA["clustering"]
    else:
        sigma = M.fit_sigma_clustering(train) if ref is None else ref["sigma_clustering"]
    emd = M.fit_emd_bandwidths(train, include_orbits=orbit) if a.kernel == "emd" else None
    sat, saturated = None, []
    if a.saturation != "off":
        sat = M.saturation(test, a.kernel, sigma, emd, include_orbits=orbit)
        saturated = [s for s, v in sat.items() if v >= M.SATURATION_THRESHOLD]
        print("[score] 1 - kbar: " + "  ".join(f"{k} {v:.3f}" for k, v in sat.items()), flush=True)
    real = M.Reference(test)
    draws = []
    for p in files:
        r = M.score_set(real, M.lcc(M.load_samples(p)), sigma, orbit=orbit, kernel=a.kernel, emd=emd)
        draws.append(dict(r, draw=os.path.basename(p)))
        print(f"[score] {p}: " + "  ".join(f"{k} {r[k]:.5g}" for k in M.STATS + ("c2st_acc",) if k in r) +
              f"  ({time.time() - t0:.0f} s)", flush=True)
    floor = M.floor_rows(real, train, len(test), sigma, orbit=orbit, kernel=a.kernel, emd=emd) if measure else ref["floor_draws"]
    # NOT_RANKED is the TV protocol's (orbit saturates under TV); the EMD Agg ranks all five statistics
    keep = None if a.kernel == "tv" else ()
    row = M.aggregate(draws, floor, a.name, not_ranked=saturated if a.saturation == "exclude" else keep)
    print(f"[score] {a.name}: {len(draws)} draws, kernel {a.kernel}, clustering bandwidth {sigma:.4g}, floor "
          f"{'measured' if measure else 'paper'} ({time.time() - t0:.0f} s)\n" + M.format_row(row, saturated) +
          "\n(* = not ranked, ^ = saturated)")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        write_json(dict(dataset=a.name, kernel=a.kernel, sigma_clustering=sigma, emd_bandwidths=emd, saturation=sat,
                       floor_measured=measure, draws=draws, floor_draws=floor if measure else None, row=row),
                  a.out)


if __name__ == "__main__":
    main()
