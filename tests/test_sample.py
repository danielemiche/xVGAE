"""Sampling: the z0 DDPM trains and samples; a draw has |split| graphs and repeats with its seed."""
import torch

from conftest import tiny_cfg


def test_ddpm_and_draw(tiny_loaders, tmp_path):
    from xvgae.data import train_sizes
    from xvgae.diffusion import Z0DDPM, encode_pool, train_z0_ddpm
    from xvgae.model import build_model, draw
    tr, va, te = tiny_loaders
    torch.manual_seed(0)
    m = build_model(tiny_cfg(), tr).eval()
    z0, n = encode_pool(m, tr, reps=2)
    assert z0.shape == (32, 8) and len(n) == 32
    dd = Z0DDPM(8, hidden=32, n_steps=20)
    losses, _ = train_z0_ddpm(dd, z0, torch.as_tensor(n), epochs=2, batch_size=8, out_dir=str(tmp_path), snapshots=2)
    assert len(losses) == 2 and (tmp_path / "z0_ddpm_ep2.pt").exists()
    dd.eval()
    ddpm = (dd, z0.mean(0), z0.std(0))
    hist = train_sizes(tr)
    g1, g2 = draw(m, te, hist, 100, ddpm=ddpm), draw(m, te, hist, 100, ddpm=ddpm)
    assert len(g1) == len(te.dataset) and [sorted(g.edges()) for g in g1] == [sorted(g.edges()) for g in g2]
    assert all(g.number_of_nodes() in set(hist.tolist()) for g in g1)
    assert len(draw(m, te, hist, 101)) == len(te.dataset)
