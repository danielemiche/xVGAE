"""Evaluation: graph statistics, MMD^2 (Gaussian kernel on TV or EMD), WL, C2ST, the train-vs-test floor and Agg.

Per sample set (|test| graphs, both sides reduced to their largest connected component): biased MMD^2 of degree,
clustering, spectral, orbit and WL against the test split, and C2ST accuracy (logistic regression on random-GIN
features). Ratio = MMD^2 / floor, floor = the mean over 5 random training subsets of |test| graphs. A row is the median
over 3 draws; Agg = the geometric mean of the median ratios over the ranked statistics.
"""
import hashlib
import json
import os
import subprocess
import tempfile
import warnings
from contextlib import contextmanager

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

STATS = ("degree", "clustering", "spectral", "orbit", "wl")
NOT_RANKED = {"orbit": {"citeseer_ego", "mmsbm_medium", "lfr_v2"}}     # orbit saturates on these corpora
PUBLISHED_SIGMA = {"degree": 1.0, "clustering": 0.1, "spectral": 1.0, "orbit": 30.0}
SPECTRAL_RANGE = (0.0, 2.0)
SATURATION_THRESHOLD = 0.9
MAX_BINS = 20000
ORCA = os.environ.get("XVGAE_ORCA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "orca"))
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_reference.json")


# ------------------------------------------------------------------------------------------------ graph statistics
@contextmanager
def one_thread():
    """torch and BLAS on one thread (statistics and C2ST that do not depend on the thread count)."""
    from threadpoolctl import threadpool_limits
    prev = torch.get_num_threads()
    try:
        with threadpool_limits(1):
            torch.set_num_threads(1)
            yield
    finally:
        torch.set_num_threads(prev)


def _map(fn, graphs):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return [fn(g) for g in graphs]


def _clustering(G):
    """Per-node clustering coefficients (equal to nx.clustering)."""
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64, weight=None)
    if A.diagonal().any():
        A.setdiag(0)
        A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel()
    tri = np.asarray((A @ A).multiply(A).sum(1)).ravel()
    den = deg * (deg - 1)
    return np.divide(tri, den, out=np.zeros_like(tri), where=den > 0)


def _spectrum(G):
    ev = np.sort(np.linalg.eigvalsh(np.asarray(nx.normalized_laplacian_matrix(G).todense())))[::-1]
    return np.round(ev, 12) + 0.0


def degree(graphs):
    return _map(lambda G: np.array([d for _, d in G.degree()], dtype=np.float64), graphs)


def clustering(graphs):
    return _map(_clustering, graphs)


def spectral(graphs):
    return _map(_spectrum, graphs)


def orbit_available():
    return os.access(os.path.join(ORCA, "orca"), os.X_OK)


def orbit_counts(graphs):
    """4-node orbit counts per node with ORCA (Hocevar & Demsar 2014): one [n, 15] array per graph."""
    orca = os.path.join(ORCA, "orca")
    if not orbit_available():
        raise FileNotFoundError(f"no ORCA binary at {orca}: run scripts/build_orca.sh")
    out = []
    for G in graphs:
        G = nx.relabel_nodes(G, {v: i for i, v in enumerate(G.nodes())})
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        n, m = G.number_of_nodes(), G.number_of_edges()
        if m == 0:
            out.append(np.zeros((max(n, 1), 15)))
            continue
        with tempfile.TemporaryDirectory() as d:
            fi, fo = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
            with open(fi, "w") as f:
                f.write(f"{n} {m}\n" + "".join(f"{u} {v}\n" for u, v in G.edges()))
            r = subprocess.run([orca, "node", "4", fi, fo], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0 or not os.path.exists(fo) or os.path.getsize(fo) == 0:
                raise RuntimeError(f"ORCA failed on a graph with n={n}, m={m}: {r.stderr[:300]}")
            orbits = np.loadtxt(fo, ndmin=2)
        if orbits.shape != (n, 15):
            raise RuntimeError(f"ORCA returned shape {orbits.shape} for n={n}")
        out.append(orbits)
    return out


def _wl_row(G, num_iter=4, dim=16384):
    """Weisfeiler-Lehman subtree colour counts (initial colour = degree, blake2b hashes) folded into dim buckets."""
    h = lambda s: int(hashlib.blake2b(s.encode("utf-8"), digest_size=4).hexdigest(), 16)  # noqa: E731
    colors = {v: int(d) for v, d in G.degree()}
    row = {}
    for it in range(num_iter + 1):
        for c in colors.values():
            row[c % dim] = row.get(c % dim, 0) + 1
        if it < num_iter:
            colors = {v: h(str((colors[v], sorted(colors[u] for u in G.neighbors(v))))) for v in G.nodes()}
    return row


def wl_features(graphs, dim=16384):
    """[graphs, dim] L2-normalised WL features (Shervashidze et al. 2011)."""
    feat = np.zeros((len(graphs), dim), dtype=np.float64)
    for i, r in enumerate(_map(_wl_row, graphs)):
        for c, cnt in r.items():
            feat[i, c] = cnt
    return feat / np.clip(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12, None)


def wl_mmd(fa, fb):
    """Biased MMD^2 with the linear kernel on WL features."""
    v = (fa @ fa.T).sum() / fa.shape[0] ** 2 + (fb @ fb.T).sum() / fb.shape[0] ** 2 - 2.0 * (fa @ fb.T).mean()
    return float(max(v, 0.0))


# ------------------------------------------------------------------------------------------------ kernels and MMD
def histograms(stats_a, stats_b, range_mode, bins=100):
    """Both sets as pmfs on one grid and the bin width. range_mode: 'auto_integer' (one bin per integer), a float w
    (fixed width w from 0) or a (lo, hi) range with `bins` bins."""
    if range_mode == "auto_integer" or isinstance(range_mode, float):
        w = 1.0 if range_mode == "auto_integer" else float(range_mode)
        top = max([float(np.max(v)) for v in list(stats_a) + list(stats_b) if len(v)] + [0.0])
        n_bins = int(min(np.floor(top / w) + 1, MAX_BINS))
        lim = (0.0, w * n_bins)
    else:
        lim, n_bins = (float(range_mode[0]), float(range_mode[1])), bins
        w = (lim[1] - lim[0]) / bins

    def pmf(stats):
        out = []
        for s in stats:
            c, _ = np.histogram(np.clip(np.asarray(s, np.float64), lim[0], lim[1]), bins=n_bins, range=lim)
            out.append(c.astype(np.float64) / c.sum() if c.sum() > 0 else np.zeros_like(c, dtype=np.float64))
        return np.array(out)
    return pmf(stats_a), pmf(stats_b), w


def distance(X, Y, kernel="tv"):
    """tv: 0.5 L1 between pmfs; emd: the exact 1-D earth mover's distance in bins (an empty row costs bins - 1)."""
    if kernel == "tv":
        return 0.5 * cdist(X, Y, metric="cityblock")
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    D = cdist(np.cumsum(X, 1), np.cumsum(Y, 1), metric="cityblock")
    mx, my = X.sum(1), Y.sum(1)
    zx, zy = mx == 0, my == 0
    if zx.any() or zy.any():
        D[np.ix_(zx, ~zy)] = (X.shape[1] - 1) * my[~zy][None, :]
        D[np.ix_(~zx, zy)] = (X.shape[1] - 1) * mx[~zx][:, None]
    return D


def gram(X, Y, sigma, kernel="tv"):
    d = distance(X, Y, kernel)
    return np.exp(-d * d / (2.0 * sigma * sigma))


def mmd2(X, Y, sigma, kernel="tv"):
    """Biased MMD^2 between the rows of X and Y (clipped at 0)."""
    v = gram(X, X, sigma, kernel).sum() / X.shape[0] ** 2 + gram(Y, Y, sigma, kernel).sum() / Y.shape[0] ** 2 \
        - 2.0 * gram(X, Y, sigma, kernel).mean()
    return float(max(v, 0.0))


def one_minus_kbar(X, sigma, kernel="tv"):
    """1 - mean kernel value between two different rows of X (1 = the kernel matrix is the identity: saturated)."""
    K = gram(X, X, sigma, kernel)
    return float(1.0 - (K.sum() - np.trace(K)) / max(K.shape[0] * (K.shape[0] - 1), 1))


def hist_mmd(stats_a, stats_b, sigma, range_mode, kernel="tv", bins=100):
    """MMD^2 of one histogram statistic; for emd, sigma is in value units."""
    X, Y, w = histograms(stats_a, stats_b, range_mode, bins)
    return mmd2(X, Y, sigma if kernel == "tv" else sigma / w, kernel)


def _mean_orbits(orbs):
    return np.asarray([g.sum(axis=0) / max(g.shape[0], 1) for g in orbs], dtype=np.float64)


def _orbit_k(orbits, k):
    return [o[:, k] for o in orbits]


HIST = {"degree": (degree, "auto_integer"), "clustering": (clustering, (0, 1)), "spectral": (spectral, SPECTRAL_RANGE)}


def val_stats(graphs, stats=("degree", "clustering", "spectral", "wl")):
    """Per-graph statistics of a reference set for the validation MMD (no LCC)."""
    return {s: wl_features(graphs) if s == "wl" else HIST[s][0](graphs) for s in stats}


def val_mmd(real, graphs):
    """Validation MMD^2 of `graphs` against val_stats `real`, published bandwidths (clustering 0.1)."""
    return {s: wl_mmd(v, wl_features(graphs)) if s == "wl" else
            hist_mmd(v, HIST[s][0](graphs), PUBLISHED_SIGMA[s], HIST[s][1]) for s, v in real.items()}


def fit_emd_bandwidths(train_graphs, include_orbits=True, seed=0):
    """EMD bandwidths from training graphs: the median positive EMD between pairs of up to 400 of them; orbit histogram
    widths ceil(max_train / 100)."""
    idx = np.sort(np.random.default_rng(seed).choice(len(train_graphs), size=min(400, len(train_graphs)), replace=False))
    sub = [train_graphs[i] for i in idx]

    def med(vals, range_mode):
        H, _, w = histograms(vals, [], range_mode)
        D = distance(H, H, "emd")
        d = D[np.triu_indices_from(D, k=1)] * w
        d = d[d > 0]
        return float(np.median(d)) if d.size else float(w)

    p = {"sigma": {s: med(fn(sub), rm) for s, (fn, rm) in HIST.items()}, "orbit_width": None}
    if include_orbits:
        orb = orbit_counts(train_graphs)
        p["orbit_width"] = [None if max(float(np.max(o[:, k])) for o in orb) <= 0
                            else float(max(1, np.ceil(max(float(np.max(o[:, k])) for o in orb) / 100))) for k in range(15)]
        for k, w in enumerate(p["orbit_width"]):
            if w is not None:
                p["sigma"][f"orbit{k}"] = med(_orbit_k([orb[i] for i in idx], k), w)
    return p


def saturation(graphs, kernel="tv", sigma_clustering=0.1, emd=None, include_orbits=True):
    """{statistic: 1 - kbar} on one (reference) set under the kernel that scores it."""
    sig = dict(PUBLISHED_SIGMA, clustering=sigma_clustering)
    out = {}
    for s, (fn, rm) in HIST.items():
        X, _, w = histograms(fn(graphs), [], rm)
        out[s] = one_minus_kbar(X, sig[s] if kernel == "tv" else emd["sigma"][s] / w, kernel)
    if include_orbits:
        orbs = orbit_counts(graphs)
        if kernel == "tv":
            out["orbit"] = one_minus_kbar(_mean_orbits(orbs), PUBLISHED_SIGMA["orbit"])
        else:
            v = [one_minus_kbar(histograms(_orbit_k(orbs, k), [], w)[0], emd["sigma"][f"orbit{k}"] / w, "emd")
                 for k, w in enumerate(emd["orbit_width"]) if w is not None]
            out["orbit"] = float(np.mean(v)) if v else float("nan")
    return out


def evaluate_mmd(test_graphs, gen_graphs, sigma_clustering=0.1, include_orbits=True, kernel="tv", emd=None):
    """{degree, clustering, spectral, orbit, wl} MMD^2. tv: published bandwidths, the GraphRNN mean orbit vector;
    emd: the fitted bandwidths (fit_emd_bandwidths), 15 per-orbit histograms summed."""
    sig = dict(PUBLISHED_SIGMA, clustering=sigma_clustering)
    out = {s: hist_mmd(fn(test_graphs), fn(gen_graphs), sig[s] if kernel == "tv" else emd["sigma"][s], rm, kernel)
           for s, (fn, rm) in HIST.items()}
    if include_orbits:
        oa, ob = orbit_counts(test_graphs), orbit_counts(gen_graphs)
        if kernel == "tv":
            out["orbit"] = mmd2(_mean_orbits(oa), _mean_orbits(ob), PUBLISHED_SIGMA["orbit"])
        else:
            out["orbit"] = float(sum(hist_mmd(_orbit_k(oa, k), _orbit_k(ob, k), emd["sigma"][f"orbit{k}"], w, "emd")
                                     for k, w in enumerate(emd["orbit_width"]) if w is not None))
    out["wl"] = wl_mmd(wl_features(test_graphs), wl_features(gen_graphs))
    return out


# ------------------------------------------------------------------------------------------------ C2ST
class GINFeatures(nn.Module):
    """Random 3-layer GIN (width 35) on all-ones node features, sum pooling (Thompson et al. 2022)."""

    def __init__(self, hidden=35, layers=3):
        super().__init__()
        from torch_geometric.nn import GINConv
        self.convs = nn.ModuleList()
        self.node_proj = nn.Linear(1, hidden)
        for _ in range(layers):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))))

    @torch.no_grad()
    def features(self, graphs):
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import global_add_pool
        from torch_geometric.utils import from_networkx
        data = []
        for g in graphs:
            d = from_networkx(g)
            if hasattr(d, "weight"):
                del d.weight
            d.x = torch.ones(d.num_nodes, 1)
            data.append(d)
        out = []
        with one_thread():
            for b in DataLoader(data, batch_size=32, shuffle=False):
                x = self.node_proj(b.x.float())
                for conv in self.convs:
                    x = torch.relu(conv(x, b.edge_index))
                out.append(global_add_pool(x, b.batch).numpy())
        return np.concatenate(out, axis=0)


def c2st_acc(real, fake):
    """5-fold accuracy of a logistic regression telling real from generated features."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    X = np.vstack([real, fake])
    y = np.concatenate([np.ones(real.shape[0]), np.zeros(fake.shape[0])])
    accs = []
    with one_thread():
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
            clf = LogisticRegression(max_iter=2000, n_jobs=1, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(X[tr], y[tr])
            accs.append(clf.score(X[te], y[te]))
    return float(np.asarray(accs).mean())


# ------------------------------------------------------------------------------------------------ the protocol
def lcc(graphs):
    """Each graph's largest connected component, relabelled 0..n-1."""
    return [G.copy() if G.number_of_nodes() == 0 else
            nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
            for G in graphs]


def pyg_to_nx(loader_or_list):
    """PyG graphs as networkx graphs (through the dense adjacency, diagonal zeroed)."""
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_dense_adj
    loader = loader_or_list if hasattr(loader_or_list, "dataset") else DataLoader(loader_or_list, batch_size=8)
    out = []
    for batch in loader:
        ei, bv = batch.edge_index, batch.batch
        for g in range(int(bv.max().item()) + 1):
            nodes = (bv == g).nonzero(as_tuple=True)[0]
            adj = to_dense_adj(ei[:, bv[ei[0]] == g] - nodes.min().item(), max_num_nodes=nodes.size(0)).squeeze(0).cpu()
            adj.fill_diagonal_(0)
            out.append(nx.from_numpy_array(adj.numpy().astype(np.float64)))
    return out


def real_split(ds, split):
    """The reference graphs of a split as scored (each graph's LCC)."""
    from xvgae.data import get_dataloaders
    loaders = dict(zip(("train", "val", "test"), get_dataloaders(ds, 8, shuffle_train=False)))
    return lcc(pyg_to_nx(loaders[split]))


def write_npz(graphs, path):
    """Graphs as one npz: sizes [G], offsets [G + 1], edges [2, E]."""
    per, offs = [], [0]
    for g in graphs:
        idx = {v: i for i, v in enumerate(g.nodes())}
        e = np.array([(idx[u], idx[v]) for u, v in g.edges()], dtype=np.int32).reshape(-1, 2).T
        per.append(e)
        offs.append(offs[-1] + e.shape[1])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, sizes=np.array([g.number_of_nodes() for g in graphs], dtype=np.int64),
                        offsets=np.array(offs, dtype=np.int64), edges=np.concatenate(per, axis=1).astype(np.int32))
    return path


def load_samples(path):
    """The graphs of an npz written by write_npz."""
    z = np.load(path, allow_pickle=False)
    out = []
    for i in range(len(z["sizes"])):
        g = nx.Graph()
        g.add_nodes_from(range(int(z["sizes"][i])))
        e = z["edges"][:, z["offsets"][i]:z["offsets"][i + 1]]
        g.add_edges_from(zip(e[0].tolist(), e[1].tolist()))
        out.append(g)
    return out


class Reference:
    """The reference set and its C2ST features (the GIN is seeded: torch.manual_seed(42))."""

    def __init__(self, graphs):
        torch.manual_seed(42)
        self.graphs, self.gin = graphs, GINFeatures()
        self.features = self.gin.features(graphs)


def score_set(ref, gen, sigma_clustering, orbit=True, kernel="tv", emd=None):
    """{degree, clustering, spectral, orbit, wl, c2st_acc, n_mean, m_mean} of one LCC sample set against `ref`."""
    out = evaluate_mmd(ref.graphs, gen, sigma_clustering, include_orbits=orbit, kernel=kernel, emd=emd)
    out["c2st_acc"] = c2st_acc(ref.features, ref.gin.features(gen))
    out["n_mean"] = float(np.mean([g.number_of_nodes() for g in gen]))
    out["m_mean"] = float(np.mean([g.number_of_edges() for g in gen]))
    return {k: float(v) for k, v in out.items()}


def fit_sigma_clustering(train):
    """Clustering bandwidth = median TV distance between the clustering histograms of pairs of training graphs."""
    H = np.array([np.histogram(np.clip(np.asarray(v, float), 0, 1), bins=100, range=(0, 1))[0] for v in clustering(train)],
                 float)
    H = H / np.clip(H.sum(1, keepdims=True), 1, None)
    tv = 0.5 * np.abs(H[:, None, :] - H[None, :, :]).sum(-1)
    return float(np.median(tv[np.triu_indices(len(H), 1)]))


def floor_rows(ref, train, n_test, sigma_clustering, seeds=5, orbit=True, kernel="tv", emd=None):
    """The floor draws: for seed s, a random subset of |test| training graphs scored like a model."""
    rows = []
    for s in range(seeds):
        idx = np.sort(np.random.default_rng(s).choice(len(train), size=min(n_test, len(train)), replace=False))
        rows.append(dict(score_set(ref, [train[i] for i in idx], sigma_clustering, orbit, kernel, emd), seed=s))
    return rows


def aggregate(draws, floor, ds, not_ranked=None):
    """Table row: per statistic the median ratio over the draws and its range, C2ST_delta = |acc - 0.5|, and Agg over the
    ranked statistics (default: all but NOT_RANKED for this corpus)."""
    stats = [s for s in STATS if all(s in r for r in draws) and all(s in r for r in floor)]
    fl = {s: float(np.mean([float(r[s]) for r in floor])) for s in stats}
    row = {"n_draws": len(draws), "floor": fl}
    for s in stats:
        v = [float(r[s]) for r in draws]
        row.update({f"mmd_{s}": float(np.median(v)), f"ratio_{s}": float(np.median(v)) / fl[s],
                    f"ratio_{s}_min": min(v) / fl[s], f"ratio_{s}_max": max(v) / fl[s]})
    c = [abs(float(r["c2st_acc"]) - 0.5) for r in draws]
    row.update(c2st_delta=float(np.median(c)), c2st_delta_min=min(c), c2st_delta_max=max(c))
    skip = {s for s, dss in NOT_RANKED.items() if ds in dss} if not_ranked is None else set(not_ranked)
    row["ranked"] = rs = [s for s in stats if s not in skip]
    row["agg"] = float(np.exp(sum(np.log(row[f"ratio_{s}"]) for s in rs) / len(rs)))
    per = [float(np.exp(np.mean([np.log(float(r[s]) / fl[s]) for s in rs]))) for r in draws]
    row.update(agg_per_draw_median=float(np.median(per)), agg_per_draw_min=min(per), agg_per_draw_max=max(per))
    return row


def paper_reference(ds):
    """Scoring references of a built-in dataset (clustering bandwidth, floor draws), else None."""
    return json.load(open(REFERENCE))["scoring"].get(ds)


def format_row(row, saturated=()):
    """Per statistic the median ratio [min-max], then Agg and C2ST_delta (* not ranked, ^ saturated)."""
    stats = [s for s in STATS if f"ratio_{s}" in row]
    cols = [s + ("" if s in row["ranked"] else "*") + ("^" if s in saturated else "") for s in stats]
    head = "  ".join(f"{c:>20}" for c in cols) + f"  {'Agg':>8}  {'C2ST_d':>14}"
    vals = "  ".join(f"{row[f'ratio_{s}']:7.2f} [{row[f'ratio_{s}_min']:5.2f}-{row[f'ratio_{s}_max']:5.2f}]" for s in stats)
    vals += f"  {row['agg']:8.3f}  {row['c2st_delta']:.3f} [{row['c2st_delta_min']:.3f}-{row['c2st_delta_max']:.3f}]"
    return head + "\n" + vals


class Selector:
    """The validation selection rule. A candidate draws a few validation sets of |val| graphs (one per seed); each set is
    scored against the validation split (MMD^2 with the EMD kernel and the fitted bandwidths, all five statistics;
    kernel="tv" gives the published kernels and ranked statistics, and the C2ST accuracy), and the MMD^2 and the accuracy
    are averaged over the sets. The lowest geometric mean of the averaged MMD^2 holds the pick. A candidate within 10 % of it
    takes the pick only when its C2ST is closer to 0.5 beyond noise: d = |acc - 0.5|, set to 0 within 2 null SDs of 0.5,
    lower than the holder's d by more than 2 SE of the difference (each SE from that candidate's per-set accuracies).
    A candidate with an averaged MMD^2 of 0 (clipped: the TV / EMD Gaussian kernels are not positive definite) has no
    geometric mean and is left out."""

    TIE_BAND, NOISE_SE = 0.10, 2.0

    def __init__(self, ds, kernel="emd"):
        ref, train, val = paper_reference(ds), real_split(ds, "train"), real_split(ds, "val")
        self.sigma = ref["sigma_clustering"] if ref else fit_sigma_clustering(train)
        self.orbit = bool(ref) or orbit_available()
        self.kernel, self.dataset = kernel, ds
        self.emd = fit_emd_bandwidths(train, include_orbits=self.orbit) if kernel == "emd" else None
        self.ranked = [s for s in STATS if (kernel == "emd" or ds not in NOT_RANKED.get(s, ())) and (self.orbit or s != "orbit")]
        self.val = Reference(val)
        # null SD of the validation C2ST: the larger of the measured one (real train subsets vs real val, 10 seeds; paper
        # corpora) and the binomial SE of a chance accuracy on 2 |val| graphs
        null = json.load(open(REFERENCE)).get("c2st_null_sd", {}).get(ds, 0.0)
        self.se0 = float(max(null, 0.5 / np.sqrt(2 * len(val))))
        self.rule = dict(kernel=kernel, statistics=self.ranked, graphs_per_set=len(val), tie_band=self.TIE_BAND,
                         noise_se=self.NOISE_SE, c2st_null_sd=self.se0, sigma_clustering=self.sigma, emd_bandwidths=self.emd)

    def score(self, draws):
        """draws: one list of graphs per seed, |val| graphs each."""
        recs = [score_set(self.val, lcc(g), self.sigma, orbit=self.orbit, kernel=self.kernel, emd=self.emd) for g in draws]
        r = {k: float(np.mean([x[k] for x in recs])) for k in recs[0]}
        acc = [x["c2st_acc"] for x in recs]
        r["c2st_se"] = float(np.std(acc, ddof=1) / np.sqrt(len(acc))) if len(acc) > 1 else 0.0
        r["clipped"] = [k for k in self.ranked if not r[k] > 0]
        r["geomean"] = float("nan") if r["clipped"] else float(np.exp(np.mean([np.log(r[k]) for k in self.ranked])))
        r["sets"] = recs
        return r

    def pick(self, res):
        """(winner, tied, d) of {candidate: score dict}."""
        ok = {k: v for k, v in res.items() if np.isfinite(v["geomean"])}
        if len(ok) < len(res):
            print(f"[select] left out, an MMD^2 clipped to 0: {sorted(set(res) - set(ok))}", flush=True)
        if not ok:
            raise RuntimeError("no candidate has a finite validation geometric mean")
        low = min(ok, key=lambda k: ok[k]["geomean"])
        tied = [k for k, v in ok.items() if v["geomean"] <= (1 + self.TIE_BAND) * ok[low]["geomean"]]
        d = {k: (lambda x: 0.0 if x <= self.NOISE_SE * self.se0 else x)(abs(ok[k]["c2st_acc"] - 0.5)) for k in tied}
        beats = lambda a, b: d[a] - d[b] > self.NOISE_SE * float(np.hypot(ok[a]["c2st_se"], ok[b]["c2st_se"]))  # noqa: E731
        better = [k for k in tied if k != low and beats(low, k)]
        return (min(better, key=lambda k: (d[k], ok[k]["geomean"])) if better else low), tied, d
