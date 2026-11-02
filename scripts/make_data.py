"""Build a corpus: a paper corpus (edge-checked), the SBM-2 example, or your own graphs.

    python scripts/make_data.py mmsbm_medium --out data       # also citeseer_ego, lfr_v2, protein
    python scripts/make_data.py sbm2 --out data               # 400 two-block SBM graphs, n 40-120
    python scripts/make_data.py custom --graphs my_graphs.pkl --name mine --out data
"""
import argparse

from xvgae import set_threads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", choices=["citeseer_ego", "mmsbm_medium", "lfr_v2", "protein", "sbm2", "custom"])
    ap.add_argument("--out", default="data", help="data root")
    ap.add_argument("--graphs", help="custom: a .pkl / .pt list of networkx or PyG graphs, or a {train, val, test} dict")
    ap.add_argument("--name", help="custom: the corpus name")
    ap.add_argument("--k", default="auto", help="eigenvectors per node (default: the smallest n, between 8 and 50)")
    ap.add_argument("--lcc", action="store_true", help="keep each graph's largest connected component")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    threads = set_threads(a.threads)
    from xvgae import data
    if a.key in ("sbm2", "custom"):
        if a.key == "custom" and not (a.graphs and a.name):
            ap.error("custom needs --graphs and --name")
        graphs = data.sbm_graphs() if a.key == "sbm2" else data.read_graphs(a.graphs)
        name = a.name or a.key
        data.build(graphs, name, out=a.out, k=a.k, lcc=a.lcc)
    else:
        from xvgae.corpora import BUILD
        BUILD[a.key](a.out, **({"workers": threads} if a.key == "lfr_v2" else {}))
        name = a.key
    for s in ("train", "val", "test"):
        data.load_split(name, s, root=a.out)
    print(f"[make_data] {name}: done" + (" (edges identical to the paper corpus)" if name in data.PAPER_CORPORA else ""))


if __name__ == "__main__":
    main()
