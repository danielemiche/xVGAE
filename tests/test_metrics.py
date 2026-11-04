"""Metrics without data: MMD and EMD properties, saturation, C2ST, the paper Agg from the frozen draws, sample files."""
import networkx as nx
import numpy as np

from xvgae import metrics as M


def er(n, p, seeds):
    return [nx.gnp_random_graph(n, p, seed=s) for s in seeds]


def test_mmd_zero_on_identical_sets_and_clustering_matches_networkx():
    gs = er(30, 0.2, range(10))
    d = M.degree(gs)
    assert M.hist_mmd(d, d, 1.0, "auto_integer") == 0.0
    assert M.hist_mmd(d, M.degree(er(30, 0.6, range(10))), 1.0, "auto_integer") > 0.0
    c = nx.clustering(gs[0])
    assert np.allclose(M.clustering(gs[:1])[0], [c[v] for v in gs[0].nodes()], atol=0)


def test_emd_and_tv_distances():
    X, Y = np.eye(4)[[0]], np.eye(4)[[1, 3]]
    assert np.allclose(M.distance(X, Y, "emd"), [[1.0, 3.0]]) and np.allclose(M.distance(X, Y, "tv"), [[1.0, 1.0]])
    ref, near, far = (np.tile(np.eye(6)[i], (5, 1)) for i in (0, 1, 5))
    assert M.mmd2(ref, far, 2.0, "emd") > M.mmd2(ref, near, 2.0, "emd") > 0.0
    assert np.isclose(M.mmd2(ref, far, 0.5), M.mmd2(ref, near, 0.5))


def test_saturation_and_emd_bandwidths():
    assert M.one_minus_kbar(np.tile(np.eye(10)[3], (6, 1)), 0.1) == 0.0
    assert M.one_minus_kbar(np.eye(10)[:6], 0.1) > 0.99 > 0.9 > M.one_minus_kbar(np.eye(10)[:6], 10.0)
    train = er(25, 0.2, range(12))
    p = M.fit_emd_bandwidths(train, include_orbits=False)
    assert set(p["sigma"]) == {"degree", "clustering", "spectral"} and all(v > 0 for v in p["sigma"].values())
    kw = dict(include_orbits=False, kernel="emd", emd=p)
    near = M.evaluate_mmd(er(25, 0.2, range(100, 106)), train[:6], **kw)
    far = M.evaluate_mmd(er(25, 0.2, range(100, 106)), er(25, 0.6, range(200, 206)), **kw)
    assert far["degree"] > near["degree"] and far["spectral"] > near["spectral"]


def test_c2st_and_score_set():
    ref = M.Reference(er(20, 0.2, range(20)))
    same = M.score_set(ref, er(20, 0.2, range(100, 120)), 0.1, orbit=False)
    other = M.score_set(ref, er(20, 0.7, range(50, 70)), 0.1, orbit=False)
    assert other["c2st_acc"] > max(0.9, same["c2st_acc"]) and other["degree"] > same["degree"]


def test_npz_round_trip(tmp_path):
    gs = [nx.gnp_random_graph(n, 0.3, seed=n) for n in (5, 9, 12)]
    back = M.load_samples(M.write_npz(gs, str(tmp_path / "s.npz")))
    assert [sorted(g.edges()) for g in back] == [sorted(g.edges()) for g in gs] and [len(g) for g in back] == [5, 9, 12]


def _selector(se0=0.02):
    sel = object.__new__(M.Selector)      # the pick rule alone, without data
    sel.se0 = se0
    return sel


def test_selector_pick_keeps_lowest_geomean_unless_c2st_beats_it_beyond_noise():
    sel = _selector()
    c = lambda g, acc, se=0.01: dict(geomean=g, c2st_acc=acc, c2st_se=se)  # noqa: E731
    # both at chance (|acc - .5| <= 2 se0): the lowest geomean holds the pick
    assert sel.pick({"a": c(1.00, 0.53), "b": c(1.05, 0.50)})[0] == "a"
    # b is tied and indistinguishable from real while a is far from chance: b takes the pick
    assert sel.pick({"a": c(1.00, 0.70), "b": c(1.05, 0.51)})[0] == "b"
    # outside the 10 % band the C2ST does not count
    assert sel.pick({"a": c(1.00, 0.70), "b": c(1.20, 0.50)})[0] == "a"
    # a gap smaller than 2 SE of the difference does not move the pick
    assert sel.pick({"a": c(1.00, 0.60, 0.05), "b": c(1.05, 0.55, 0.05)})[0] == "a"
    # a clipped MMD^2 (geomean nan) is left out
    assert sel.pick({"a": c(float("nan"), 0.5), "b": c(2.0, 0.5)})[0] == "b"
