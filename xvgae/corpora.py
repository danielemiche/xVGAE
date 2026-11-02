"""The four paper corpora, rebuilt exactly (edges checked against paper_reference.json by xvgae.data.load_split).

citeseer_ego  3-hop ego graphs of CiteSeer with 50-400 nodes (Planetoid download), 484 / 121 / 152, k = 50
mmsbm_medium  mixed-membership SBM, n 120-400, K in {3, 5}, seed 4242, 640 / 160 / 200, k = 100
lfr_v2        LFR benchmark with every parameter drawn per graph, 640 / 160 + a fresh test set of 200, k = 20
protein       D&D graphs with 100-500 nodes (TUDataset download), 587 / 147 / 184, k = 100
"""
import multiprocessing as mp
import os
import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph  # noqa: F401
import scipy.sparse.linalg as spla
import torch
from torch_geometric.data import Data


def _save(out_root, name, splits):
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    for s, graphs in splits.items():
        torch.save(graphs, os.path.join(d, f"{name}_{s}.pt"))
    print(f"[{name}] " + " / ".join(f"{len(g)}" for g in splits.values()) + f" graphs -> {d}", flush=True)
    return d


def _sign_fix(vecs, k):
    for c in range(k):
        j = int(np.argmax(np.abs(vecs[:, c])))
        if vecs[j, c] < 0:
            vecs[:, c] = -vecs[:, c]
    return vecs


# ------------------------------------------------------------------------------------------------ CiteSeer-Ego
def _ego_features(A, d=50):
    """(eigvals [1, d], eigvecs [n, d]) of the normalised Laplacian in float32, largest |entry| per column positive."""
    n = A.shape[0]
    A = A.float().unsqueeze(0)
    k = min(d, n)
    inv = 1 / torch.sqrt(A.sum(-1, keepdim=True).clamp_min(1e-8))
    L = torch.eye(n, dtype=A.dtype).unsqueeze(0).expand(1, -1, -1) - A * inv.transpose(1, 2) * inv
    vals, vecs = torch.linalg.eigh(L)
    vals, vecs = vals[:, :k], vecs[:, :, :k]
    picked = vecs[torch.arange(1)[:, None].expand(-1, k), vecs.abs().argmax(dim=1), torch.arange(k)[None, :]]
    signs = torch.sign(picked).unsqueeze(1)
    signs[signs == 0] = 1.0
    vals, vecs = vals[0], (vecs * signs)[0]
    if k < d:
        vals, vecs = torch.cat([vals, torch.zeros(d - k)]), torch.cat([vecs, torch.zeros(n, d - k)], dim=1)
    return vals.unsqueeze(0), vecs


def citeseer_ego(out_root):
    from torch_geometric.datasets import Planetoid
    cache = os.path.join(out_root, "_downloads")
    Planetoid(root=cache, name="CiteSeer")
    with open(os.path.join(cache, "CiteSeer", "raw", "ind.citeseer.graph"), "rb") as f:
        G = nx.from_dict_of_lists(pkl.load(f, encoding="latin1"))
    G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
    items = []
    for i in range(G.number_of_nodes()):
        ego = nx.ego_graph(G, i, radius=3)
        if not 50 <= ego.number_of_nodes() <= 400:
            continue
        mapping = {i: 0, **{old: new for new, old in enumerate(sorted(v for v in ego.nodes() if v != i), start=1)}}
        g = nx.relabel_nodes(ego, mapping)
        A = torch.from_numpy(nx.to_numpy_array(g, nodelist=sorted(g.nodes()), dtype=np.float32))
        A.fill_diagonal_(0)
        src, dst = (A > 0).nonzero(as_tuple=True)
        ev, x = _ego_features(A)
        items.append(Data(x=x.view(-1, 50), eigvals=ev.float(), eigvecs=x, edge_index=torch.stack([src, dst]).long(),
                          num_nodes=g.number_of_nodes()))
    idx = np.arange(len(items))
    np.random.RandomState(42123).shuffle(idx)
    n_tv = int(0.8 * len(items))
    n_tr = int(0.8 * n_tv)
    pick = lambda ix: [items[i] for i in ix]  # noqa: E731
    return _save(out_root, "citeseer_ego", {"train": pick(idx[:n_tr]), "val": pick(idx[n_tr:n_tv]), "test": pick(idx[n_tv:])})


# ------------------------------------------------------------------------------------------------ MMSBM
def _mmsbm_graph(rng):
    """One MMSBM draw reduced to its largest component (None if it has fewer than 120 nodes)."""
    var = 60.0 ** 2
    p = 250.0 / var
    N = int(np.clip(int(rng.negative_binomial(250.0 * p / (1 - p), p)), 120, 400))
    K = int(rng.choice((3, 5)))
    alpha = max(rng.gamma(2.0, 0.15), 1e-3)
    p_in = float(np.clip(rng.beta(50.0 * 0.5, 50.0 * (1 - 0.5)), 1e-4, 1 - 1e-4))
    p_out = float(np.clip(rng.beta(50.0 * 0.025, 50.0 * (1 - 0.025)), 1e-5, 1 - 1e-5))
    pi = rng.dirichlet(np.maximum(alpha * np.ones(K), 1e-2))
    B = np.empty((K, K))
    for k in range(K):
        B[k, k] = float(np.clip(rng.beta(50.0 * p_in, 50.0 * (1 - p_in)), 1e-4, 1 - 1e-4))
        for m in range(k + 1, K):
            B[k, m] = B[m, k] = float(np.clip(rng.beta(50.0 * p_out, 50.0 * (1 - p_out)), 1e-5, 1 - 1e-5))
    cdf = np.cumsum(rng.dirichlet(np.maximum(alpha * pi, 1e-2), size=N), axis=1)
    iu, ju = np.triu_indices(N, k=1)
    k_ij = (cdf[iu] >= rng.uniform(size=iu.size)[:, None]).argmax(axis=1)
    k_ji = (cdf[ju] >= rng.uniform(size=iu.size)[:, None]).argmax(axis=1)
    accept = rng.uniform(size=iu.size) < B[k_ij, k_ji]
    rows, cols = iu[accept], ju[accept]
    if rows.size == 0:
        return None
    ij = (np.concatenate([rows, cols]), np.concatenate([cols, rows]))
    A = sp.coo_matrix((np.ones(2 * rows.size, dtype=np.float32), ij), shape=(N, N), dtype=np.float32).tocsr()
    n_comp, labels = sp.csgraph.connected_components(A, directed=False)
    if n_comp > 1:
        keep = np.where(labels == int(np.argmax(np.bincount(labels))))[0]
        if keep.size < 120:
            return None
        A = A[keep][:, keep]
    return A


def _mmsbm_data(A, d=100):
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel().clip(min=1e-12)
    Dm = sp.diags(1.0 / np.sqrt(deg))
    L = sp.eye(n, format="csr") - (Dm @ A @ Dm)
    L = (L + L.T) * 0.5
    k = min(d, n - 1) if n > 1 else 1
    ev, x = np.zeros(d, dtype=np.float32), np.zeros((n, d), dtype=np.float32)
    try:
        vals, vecs = spla.eigsh(L.astype(np.float64), k=k, which="SA", tol=1e-6, maxiter=2000)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
    except Exception:  # noqa: BLE001
        Ld = L.toarray()
        vals_full, vecs_full = np.linalg.eigh((Ld + Ld.T) * 0.5)
        vals, vecs = vals_full[:k], vecs_full[:, :k]
    ev[:k], x[:, :k] = vals.astype(np.float32), vecs.astype(np.float32)
    x = _sign_fix(x, k)
    coo = A.tocoo()
    keep = coo.row != coo.col
    xt = torch.tensor(x, dtype=torch.float32)
    return Data(x=xt, eigvals=torch.tensor(ev[None, :], dtype=torch.float32), eigvecs=xt, num_nodes=int(n),
                edge_index=torch.tensor(np.stack([coo.row[keep], coo.col[keep]], axis=0), dtype=torch.long))


def mmsbm_medium(out_root, n_graphs=1000):
    rng = np.random.default_rng(4242)
    graphs = []
    for i in range(n_graphs):
        for _ in range(20):
            A = _mmsbm_graph(rng)
            if A is not None:
                break
        else:
            raise RuntimeError("no MMSBM draw with a component of >= 120 nodes in 20 attempts")
        graphs.append(_mmsbm_data(A))
        if (i + 1) % 100 == 0:
            print(f"  [mmsbm_medium] {i + 1}/{n_graphs}", flush=True)
    idx = np.random.default_rng(4243).permutation(n_graphs)
    n_test, n_val = int(0.20 * n_graphs), int(0.16 * n_graphs)
    return _save(out_root, "mmsbm_medium", {"train": [graphs[i] for i in idx[n_test + n_val:]],
                                            "val": [graphs[i] for i in idx[n_test:n_test + n_val]],
                                            "test": [graphs[i] for i in idx[:n_test]]})


# ------------------------------------------------------------------------------------------------ LFR
class Invalid(Exception):
    """A draw that cannot be completed as a valid LFR graph (redrawn with the next seed)."""


def _powerlaw_ints(rng, lo, hi, tau, size):
    xs = np.arange(lo, hi + 1)
    p = xs.astype(float) ** -tau
    return rng.choice(xs, size=size, p=p / p.sum())


def _community_sizes(rng, n, lo, hi, tau2):
    """Power-law sizes on [lo, hi] summing exactly to n (draw until the sum reaches n, then trim)."""
    sizes = []
    while sum(sizes) < n:
        sizes.append(int(_powerlaw_ints(rng, lo, hi, tau2, 1)[0]))
    excess = sum(sizes) - n
    for i in rng.permutation(len(sizes)):
        take = min(excess, sizes[i] - lo)
        sizes[i] -= take
        excess -= take
        if excess == 0:
            break
    if excess:
        raise Invalid("community sizes")
    return np.array(sizes)


def _assign(rng, kin, sizes, max_iters):
    """A node joins a random community if its internal degree fits; a full community evicts a random member."""
    comm = -np.ones(len(kin), dtype=int)
    members = [[] for _ in sizes]
    free = list(rng.permutation(len(kin)))
    for _ in range(max_iters):
        if not free:
            return comm
        v = free.pop()
        c = int(rng.integers(len(sizes)))
        if kin[v] < sizes[c]:
            members[c].append(v)
            comm[v] = c
            if len(members[c]) > sizes[c]:
                u = members[c].pop(int(rng.integers(len(members[c]))))
                comm[u] = -1
                free.append(u)
        else:
            free.append(v)
    raise Invalid("community assignment")


def _wire(rng, stubs, ok, max_swaps):
    """Configuration model, then degree-preserving swaps until every edge satisfies ok(u, v) and none repeats."""
    E = [tuple(sorted(e)) for e in rng.permutation(stubs).reshape(-1, 2).tolist()]
    count = {}
    for e in E:
        count[e] = count.get(e, 0) + 1
    bad = lambda e: (not ok(*e)) or count[e] > 1  # noqa: E731
    todo = [i for i, e in enumerate(E) if bad(e)]
    swaps = 0
    while todo:
        i = todo[-1]
        if not bad(E[i]):
            todo.pop()
            continue
        swaps += 1
        if swaps > max_swaps:
            raise Invalid("rewiring")
        j = int(rng.integers(len(E)))
        if j == i:
            continue
        (u, v), (x, y) = E[i], E[j]
        if rng.random() < 0.5:
            x, y = y, x
        a, b = tuple(sorted((u, x))), tuple(sorted((v, y)))
        if not (ok(*a) and ok(*b)) or a == b or count.get(a, 0) or count.get(b, 0):
            continue
        for e in (E[i], E[j]):
            count[e] -= 1
            if not count[e]:
                del count[e]
        E[i], E[j] = a, b
        count[a] = count[b] = 1
        if bad(E[j]):
            todo.append(j)
    return E


def _wire_internal(rng, members, degs):
    """A simple graph with exactly these degrees: Havel-Hakimi, then degree-preserving double-edge swaps."""
    degs = [int(x) for x in degs]
    if sum(degs) == 0:
        return []
    if not nx.is_graphical(degs):
        raise Invalid("internal degrees not graphical")
    H = nx.havel_hakimi_graph(degs)
    m = H.number_of_edges()
    if m >= 2 and H.number_of_nodes() >= 4:
        try:
            nx.double_edge_swap(H, nswap=20 * m, max_tries=200 * m, seed=int(rng.integers(2**31 - 1)))
        except (nx.NetworkXAlgorithmError, nx.NetworkXError):
            pass
    return [(int(members[u]), int(members[v])) for u, v in H.edges()]


def lfr_graph(n, tau1, tau2, mu, kmin, kmax, smin, smax, rng):
    """One LFR graph (Lancichinetti, Fortunato & Radicchi 2008) with exact degrees and internal degrees, per-node mixing
    within one stub of mu, simple and connected. Returns (graph, community, k, kin)."""
    k = _powerlaw_ints(rng, kmin, kmax, tau1, n)
    if k.sum() % 2:
        i = int(rng.integers(n))
        k[i] = k[i] + 1 if k[i] < kmax else k[i] - 1
    kin0 = np.minimum(np.rint((1 - mu) * k).astype(int), k)
    for _ in range(200):
        try:
            sizes = _community_sizes(rng, n, smin, smax, tau2)
            if kin0.max() >= sizes.max():
                continue
            comm = _assign(rng, kin0, sizes, max_iters=200 * n)
        except Invalid:
            continue
        kin, ok = kin0.copy(), True
        for c in range(len(sizes)):                     # parity: move one stub where it costs least
            m = np.flatnonzero(comm == c)
            if kin[m].sum() % 2:
                target, best = (1 - mu) * k[m], None
                for alt in (kin[m] + 1, kin[m] - 1):
                    feas = (alt >= 0) & (alt <= k[m]) & (alt < sizes[c]) & (np.abs(target - alt) <= 1.0)
                    cost = np.where(feas, np.abs(target - alt), np.inf)
                    j = int(np.argmin(cost))
                    if np.isfinite(cost[j]) and (best is None or cost[j] < best[0]):
                        best = (cost[j], m[j], alt[j])
                if best is None:
                    ok = False
                    break
                kin[best[1]] = best[2]
        if ok:
            break
    else:
        raise Invalid("no community structure")
    kext = k - kin
    for _ in range(20):
        try:
            edges = []
            for c in range(len(sizes)):
                m = np.flatnonzero(comm == c)
                edges += _wire_internal(rng, m, kin[m])
            st = np.repeat(np.arange(n), kext)
            if len(st):
                edges += _wire(rng, st, lambda u, v: comm[u] != comm[v], max_swaps=50 * len(st) + 1000)
        except Invalid:
            continue
        H = nx.Graph()
        H.add_nodes_from(range(n))
        H.add_edges_from(edges)
        if nx.is_connected(H):
            break
    else:
        raise Invalid("wiring")
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    deg = np.array([G.degree(v) for v in range(n)])
    din = np.array([sum(comm[w] == comm[v] for w in G[v]) for v in range(n)])
    assert G.number_of_edges() == len(edges) and nx.number_of_selfloops(G) == 0 and nx.is_connected(G)
    assert (deg == k).all() and (din == kin).all() and (deg >= kmin).all() and (deg <= kmax).all()
    assert (np.abs(kext / k - mu) <= 1.0 / k + 1e-12).all()
    cs = np.bincount(comm)
    assert cs.min() >= smin and cs.max() <= smax and cs.sum() == n and (kin < cs[comm]).all()
    return G, comm, k, kin


def _lfr_features(G, n, dim=20):
    L = nx.normalized_laplacian_matrix(G, nodelist=range(n)).todense()
    vals, vecs = np.linalg.eigh(np.asarray(L, dtype=np.float64))
    order = np.argsort(vals)
    take = min(dim, n)
    vals, vecs = vals[order][:take], _sign_fix(vecs[:, order][:, :take], take)
    x, ev = np.zeros((n, dim), dtype=np.float32), np.zeros((1, dim), dtype=np.float32)
    x[:, :take], ev[0, :take] = vecs.astype(np.float32), vals.astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(ev)


def lfr_data(i, base_seed=20260927):
    """Graph i: the first seed >= base_seed + 1000 i whose parameters give a valid LFR graph."""
    for attempt in range(1000):
        rng = np.random.default_rng(base_seed + 1000 * i + attempt)
        n = int(rng.integers(200, 501))
        kmax = min(int(round(np.exp(rng.uniform(np.log(20), np.log(150))))), n // 2)
        kmin, tau1, tau2, mu = int(rng.integers(3, 11)), float(rng.uniform(2.0, 3.0)), float(rng.uniform(1.1, 2.0)), \
            float(rng.uniform(0.05, 0.5))
        try:
            G = lfr_graph(n, tau1, tau2, mu, kmin, kmax, 20, n // 2, rng)[0]
        except Invalid:
            continue
        x, ev = _lfr_features(G, n)
        ei = np.array([[u, v] for u, v in G.edges()] + [[v, u] for u, v in G.edges()]).T
        return Data(x=x, edge_index=torch.from_numpy(ei).long(), eigvals=ev, eigvecs=x.clone(), num_nodes=n)
    raise RuntimeError(f"LFR graph {i}: no valid draw in 1000 attempts")


def _lfr_one(args):
    # plain arrays across the pool: torch tensors would travel as shared-memory file descriptors, hours for 1200 graphs
    d = lfr_data(*args)
    return d.x.numpy(), d.edge_index.numpy(), d.eigvals.numpy(), d.num_nodes


def _lfr_rebuild(x, ei, ev, n):
    x = torch.from_numpy(x)
    return Data(x=x, edge_index=torch.from_numpy(ei), eigvals=torch.from_numpy(ev), eigvecs=x.clone(), num_nodes=n)


def _one_thread():
    from xvgae import set_threads
    set_threads(1)


def lfr_v2(out_root, workers=4, n_graphs=1000, n_test=200):
    with mp.Pool(workers, initializer=_one_thread) as pool:
        graphs = [_lfr_rebuild(*r) for r in pool.map(_lfr_one, [(i, 20260927) for i in range(n_graphs)], chunksize=8)]
        test = [_lfr_rebuild(*r) for r in pool.map(_lfr_one, [(i, 30_000_000 + 1_000_000 * 17) for i in range(n_test)],
                                                   chunksize=4)]
    perm = np.random.default_rng(20260927).permutation(n_graphs)
    n_tr, n_va = int(0.64 * n_graphs), int(0.16 * n_graphs)
    return _save(out_root, "lfr_v2", {"train": [graphs[i] for i in perm[:n_tr]],
                                      "val": [graphs[i] for i in perm[n_tr:n_tr + n_va]], "test": test})


# ------------------------------------------------------------------------------------------------ Protein (D&D)
def _protein_features(edge_index, n, d=100):
    A = torch.zeros(n, n, dtype=torch.float64)
    if edge_index.numel() > 0:
        A[edge_index[0], edge_index[1]] = 1.0
    A = ((A + A.T) > 0).double()
    deg = A.sum(dim=1)
    dinv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    L = torch.eye(n, dtype=torch.float64) - (dinv[:, None] * A * dinv[None, :])
    vals, vecs = torch.linalg.eigh(0.5 * (L + L.T))
    k = min(d, n)
    vals, vecs = vals[:k], vecs[:, :k]
    for j in range(k):
        if vecs[int(torch.argmax(vecs[:, j].abs())), j] < 0:
            vecs[:, j] = -vecs[:, j]
    if k < d:
        vecs = torch.cat([vecs, torch.zeros(n, d - k, dtype=torch.float64)], dim=1)
        vals = torch.cat([vals, torch.zeros(d - k, dtype=torch.float64)], dim=0)
    return vecs.float(), vals.float().unsqueeze(0)


def protein(out_root):
    from torch_geometric.datasets import TUDataset
    ds = TUDataset(root=os.path.join(out_root, "_downloads"), name="DD", use_node_attr=True)
    keep = [i for i, g in enumerate(ds) if 100 <= g.num_nodes <= 500]
    if len(keep) != 918:
        raise RuntimeError(f"expected 918 D&D graphs with 100 <= n <= 500, got {len(keep)}")
    perm = np.random.RandomState(42123).permutation(918)
    splits = {}
    for name, idx in (("train", perm[:587]), ("val", perm[587:734]), ("test", perm[734:])):
        splits[name] = []
        for j in idx:
            g = ds[keep[int(j)]]
            x, ev = _protein_features(g.edge_index.long(), int(g.num_nodes))
            splits[name].append(Data(x=x, eigvecs=x, eigvals=ev, edge_index=g.edge_index.long(), num_nodes=int(g.num_nodes)))
    return _save(out_root, "protein", splits)


BUILD = {"citeseer_ego": citeseer_ego, "mmsbm_medium": mmsbm_medium, "lfr_v2": lfr_v2, "protein": protein}
