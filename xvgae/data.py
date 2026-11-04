"""Data: loading the corpora (edge-checked paper corpora or your own), building your own corpus, the SBM example.

A corpus is <root>/<name>/<name>_{train,val,test}.pt, each a list of torch_geometric Data with x = eigvecs [n, k] (the k
smallest eigenvectors of the normalised Laplacian), eigvals [1, k], edge_index [2, 2m] and num_nodes (docs/DATA.md).
"""
import hashlib
import json
import os
import pickle

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

PAPER_CORPORA = ("citeseer_ego", "mmsbm_medium", "lfr_v2", "protein")
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_reference.json")


def data_root():
    return os.environ.get("XVGAE_DATA_DIR", "data")


def set_data_root(path):
    """Point every later load at `path` (the scripts' --data)."""
    if path:
        os.environ["XVGAE_DATA_DIR"] = os.path.abspath(path)


def edge_digest(graphs):
    """sha256 over the graphs (in file order) of n and the sorted undirected pairs."""
    h = hashlib.sha256()
    for g in graphs:
        ei = g.edge_index.cpu().numpy()
        a, b = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
        h.update(np.int64(int(g.num_nodes)).tobytes())
        h.update(np.unique(np.stack([a[a != b], b[a != b]], 1).astype(np.int64), axis=0).tobytes())
    return h.hexdigest()


def check_schema(graphs, where):
    """Raise a readable error when a graph does not follow the schema."""
    k = graphs[0].x.shape[1] if getattr(graphs[0], "x", None) is not None else None
    for i, g in enumerate(graphs):
        miss = [f for f in ("x", "eigvecs", "eigvals", "edge_index") if getattr(g, f, None) is None]
        if miss:
            raise ValueError(f"{where}[{i}]: missing {miss} (schema: docs/DATA.md; build it with scripts/make_data.py custom)")
        n = int(g.num_nodes)
        if g.x.shape != (n, k) or g.eigvecs.shape != g.x.shape or tuple(g.eigvals.shape) != (1, k) or \
                (g.edge_index.numel() and int(g.edge_index.max()) >= n):
            raise ValueError(f"{where}[{i}]: x {tuple(g.x.shape)}, eigvecs {tuple(g.eigvecs.shape)}, eigvals "
                             f"{tuple(g.eigvals.shape)}, n = {n}: need [n, k], [n, k], [1, k], one k per corpus, edges < n")


def load_split(name, split, root=None):
    """The graphs of one split; a built-in dataset is checked against its edge hash, your own against the schema."""
    p = os.path.join(root or data_root(), name, f"{name}_{split}.pt")
    if not os.path.exists(p):
        how = name if name in PAPER_CORPORA else f"custom --graphs <file> --name {name}"
        raise FileNotFoundError(f"{p} not found: build it with python scripts/make_data.py {how} (or point --data at it)")
    graphs = torch.load(p, weights_only=False)
    if name in PAPER_CORPORA:
        ref = json.load(open(REFERENCE))["edge_sha256"][name][split]
        if edge_digest(graphs) != ref:
            raise RuntimeError(f"{p}: its edges do not match the built-in dataset {name}/{split}; regenerate it (scripts/make_data.py)")
    else:
        check_schema(graphs, p)
    return graphs


def get_dataloaders(name, batch_size=8, shuffle_train=True, root=None):
    """(train, val, test) DataLoaders; only train is shuffled."""
    tr, va, te = (load_split(name, s, root) for s in ("train", "val", "test"))
    print(f"[data] {name}: train {len(tr)}, val {len(va)}, test {len(te)}", flush=True)
    return (DataLoader(tr, batch_size=batch_size, shuffle=shuffle_train), DataLoader(va, batch_size=batch_size),
            DataLoader(te, batch_size=batch_size))


def train_sizes(train_loader):
    """The sorted training sizes (generated graphs draw n from them)."""
    return np.sort(np.array([int(x) for b in train_loader for x in np.bincount(b.batch.numpy()).tolist() if x > 0]))


# ------------------------------------------------------------------------------------------------ your own corpus
def _as_nx(g):
    """A simple undirected networkx graph with nodes 0..n-1, from networkx or PyG."""
    if isinstance(g, Data):
        G = nx.Graph()
        G.add_nodes_from(range(int(g.num_nodes)))
        if g.edge_index is not None and g.edge_index.numel():
            G.add_edges_from(map(tuple, g.edge_index.t().tolist()))
    elif isinstance(g, nx.Graph):
        G = nx.Graph(g)
    else:
        raise TypeError(f"a graph must be a networkx.Graph or a torch_geometric Data, got {type(g).__name__}")
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return nx.convert_node_labels_to_integers(G)


def spectral_features(G, k):
    """(eigvecs [n, k], eigvals [1, k]) of the normalised Laplacian; each column's largest |entry| positive; zero-padded."""
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=np.float64, weight=None)
    deg = A.sum(1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
    L = np.eye(n) - dinv[:, None] * A * dinv[None, :]
    w, v = np.linalg.eigh(0.5 * (L + L.T))
    m = min(k, n)
    w, v = w[:m], v[:, :m].copy()
    for j in range(m):
        if v[int(np.argmax(np.abs(v[:, j]))), j] < 0:
            v[:, j] = -v[:, j]
    x, ev = np.zeros((n, k), np.float32), np.zeros((1, k), np.float32)
    x[:, :m], ev[0, :m] = v, w
    return torch.from_numpy(x), torch.from_numpy(ev)


def to_data(g, k):
    """One graph in the schema."""
    G = _as_nx(g)
    x, ev = spectral_features(G, k)
    e = np.array(list(G.edges()), dtype=np.int64).reshape(-1, 2).T
    ei = torch.from_numpy(np.concatenate([e, e[::-1]], 1).copy()) if e.size else torch.zeros(2, 0, dtype=torch.long)
    return Data(x=x, eigvecs=x, eigvals=ev, edge_index=ei, num_nodes=G.number_of_nodes())


def read_graphs(path):
    """A list (or a {train, val, test} dict) of graphs from a .pkl / .pt file, or the graphs of a sample .npz."""
    if path.endswith(".npz"):
        from xvgae.metrics import load_samples
        return load_samples(path)
    if path.endswith((".pkl", ".pickle")):
        return pickle.load(open(path, "rb"))
    return torch.load(path, weights_only=False)


def selection_floor(train_nx, val_nx, seeds=5):
    """{statistic: 5 values}: random training subsets of |val| graphs against the validation graphs, scored with the
    trainer's validation kernels (degree, clustering, spectral, WL)."""
    from xvgae.metrics import val_stats, val_mmd
    real = val_stats(val_nx)
    per = {m: [] for m in real}
    for s in range(seeds):
        idx = np.sort(np.random.default_rng(s).choice(len(train_nx), size=min(len(val_nx), len(train_nx)), replace=False))
        for m, v in val_mmd(real, [train_nx[i] for i in idx]).items():
            per[m].append(v)
    return per


def build(graphs, name, out="data", k="auto", lcc=False, fractions=(0.64, 0.16, 0.20), seed=0):
    """Write the corpus `name` to out/name/: the three splits (a seeded random split unless `graphs` is a {train, val,
    test} dict), meta.json and floor_selection.json. k = auto: the smallest graph's n, at most 50, at least 8."""
    if name in PAPER_CORPORA or not name or os.sep in name or name.startswith("."):
        raise ValueError(f"bad corpus name {name!r} (the built-in dataset names are taken)")
    if not isinstance(graphs, dict):
        graphs = list(graphs)
        idx = np.random.default_rng(seed).permutation(len(graphs))
        f = np.asarray(fractions, float) / float(np.sum(fractions))
        n_val, n_test = max(1, int(round(f[1] * len(graphs)))), max(1, int(round(f[2] * len(graphs))))
        n_tr = len(graphs) - n_val - n_test
        if n_tr < 1:
            raise ValueError(f"{len(graphs)} graphs are too few to split")
        graphs = {"train": [graphs[i] for i in idx[:n_tr]], "val": [graphs[i] for i in idx[n_tr:n_tr + n_val]],
                  "test": [graphs[i] for i in idx[n_tr + n_val:]]}
    splits = {}
    for s in ("train", "val", "test"):
        gs = [_as_nx(g) for g in graphs[s]]
        if lcc:
            from xvgae.metrics import lcc as largest_components
            gs = largest_components(gs)
        splits[s] = [G for G in gs if G.number_of_nodes() >= 2]
    sizes = [G.number_of_nodes() for gs in splits.values() for G in gs]
    k = int(min(50, max(8, min(sizes)))) if k in (None, "auto") else int(k)
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    info = {}
    for s, gs in splits.items():
        torch.save([to_data(G, k) for G in gs], os.path.join(d, f"{name}_{s}.pt"))
        ns = [G.number_of_nodes() for G in gs]
        info[s] = dict(graphs=len(gs), n_min=min(ns), n_max=max(ns))
        print(f"[data] {name}/{s}: {len(gs)} graphs, n {min(ns)}-{max(ns)}, k = {k}", flush=True)
    floor = selection_floor(splits["train"], splits["val"])
    json.dump(dict(per_seed=floor), open(os.path.join(d, "floor_selection.json"), "w"))
    json.dump(dict(name=name, k=k, lcc=bool(lcc), splits=info), open(os.path.join(d, "meta.json"), "w"), indent=1)
    return d


def sbm_graphs(graphs=400, blocks=2, n_min=40, n_max=120, p_in=0.3, p_out=0.05, seed=0):
    """Connected k-block SBM graphs (n uniform in [n_min, n_max], blocks of n // K nodes): the SBM-2 example corpus."""
    rng = np.random.default_rng(seed)
    P = [[p_in if a == b else p_out for b in range(blocks)] for a in range(blocks)]
    out = []
    while len(out) < graphs:
        n = int(rng.integers(n_min, n_max + 1))
        G = nx.stochastic_block_model([n // blocks] * (blocks - 1) + [n - (n // blocks) * (blocks - 1)], P,
                                      seed=int(rng.integers(1 << 30)))
        if nx.is_connected(G):
            H = nx.Graph()
            H.add_nodes_from(range(n))
            H.add_edges_from(G.edges())
            out.append(H)
    return out
