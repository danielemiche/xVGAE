import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tiny_cfg(decoder="sab", **kw):
    from xvgae.train import standard_config, finalise_budget
    cfg = standard_config("tiny", decoder)
    cfg.update(z_dim=4, z0_dim=8, hidden_dim=16, num_layers=2, attn_heads=2, bilinear_heads=2, batch_size=4, val_freq=1,
               val_mmd=False)
    finalise_budget(cfg, 4, epochs=4)
    return {**cfg, **kw}


@pytest.fixture
def tiny_loaders():
    """24 small connected Erdos-Renyi graphs in the corpus schema (k = 8), split 16 / 4 / 4."""
    from torch_geometric.loader import DataLoader
    from xvgae.data import to_data
    rng = np.random.default_rng(0)
    gs = []
    while len(gs) < 24:
        G = nx.gnp_random_graph(int(rng.integers(10, 18)), 0.3, seed=int(rng.integers(1 << 30)))
        if nx.is_connected(G):
            gs.append(to_data(G, 8))
    return (DataLoader(gs[:16], batch_size=4, shuffle=True), DataLoader(gs[16:20], batch_size=4),
            DataLoader(gs[20:], batch_size=4))
