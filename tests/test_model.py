"""Stage 1: forward, loss and backward for both decoders and the CiteSeer-Ego xVGAE-SAB configuration; equivariance."""
import json
import os

import pytest
import torch

from conftest import tiny_cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ego_sab_cfg():
    cfg = json.load(open(os.path.join(ROOT, "configs", "citeseer_ego_sab.json")))
    return {**cfg, "z_dim": 4, "z0_dim": 8, "hidden_dim": 16, "num_layers": 2, "attn_heads": 2, "bilinear_heads": 2}


CONFIGS = {"sab": tiny_cfg("sab"), "mlp": tiny_cfg("mlp"),
           "mlp_n_in_decoder": tiny_cfg("mlp", cond_decoder_n=True, cond_z0_prior_n=False), "ego_sab_gps": ego_sab_cfg()}


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CONFIGS.keys())
def test_forward_loss_backward_sample(tiny_loaders, cfg):
    from xvgae.model import build_model
    tr, _, _ = tiny_loaders
    torch.manual_seed(0)
    m = build_model(cfg, tr)
    b = next(iter(tr))
    out = m(b)
    loss, st = m.compute_loss(out, b, beta=1.0, beta0=0.1, free_bits_z=cfg["free_bits_z"])
    assert torch.isfinite(loss) and st["kl_z"] >= 0 and st["kl_z0"] >= 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.sample_logits([5, 9, 7])["logits_vec"].shape == (3, 9 * 8 // 2)


@pytest.mark.parametrize("decoder", ["sab", "mlp"])
def test_decoder_permutation_equivariant(tiny_loaders, decoder):
    from xvgae.model import build_model
    torch.manual_seed(0)
    m = build_model(tiny_cfg(decoder), tiny_loaders[0]).eval()
    n, perm = 9, torch.randperm(9)
    z, z0, mask = torch.randn(1, n, 4), torch.randn(1, 8), torch.ones(1, n, dtype=torch.bool)
    i, j = torch.tril_indices(n, n, -1)

    def dense(lv):
        A = torch.zeros(n, n)
        A[i, j] = lv[0]
        return A + A.T
    with torch.no_grad():
        A, Ap = dense(m.decoder(z, z0, mask=mask)), dense(m.decoder(z[:, perm], z0, mask=mask))
    assert torch.allclose(Ap, A[perm][:, perm], atol=1e-5)
