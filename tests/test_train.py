"""Training: a short run writes its checkpoints and trajectory, the validity gate reads it, the KL floor binds."""
import json

import torch

from conftest import tiny_cfg


def test_train_run_and_gate(tiny_loaders, tmp_path):
    from xvgae.model import build_model
    from xvgae.train import train, check_validity, EXIT
    tr, va, _ = tiny_loaders
    torch.manual_seed(0)
    train(build_model(tiny_cfg(), tr), tr, va, tiny_cfg(), str(tmp_path))
    for f in ("config.json", "model_last.pt", "model_best_warmup.pt", "trajectory_local.json", "run_meta.json"):
        assert (tmp_path / f).exists(), f
    assert json.load(open(tmp_path / "run_meta.json"))["last_epoch"] == 3
    assert check_validity(str(tmp_path), quiet=True)["kind"] in EXIT


def test_kl_floor_stops_the_kl_gradient(tiny_loaders):
    from xvgae.model import build_model
    torch.manual_seed(0)
    m = build_model(tiny_cfg(), tiny_loaders[0])
    b = next(iter(tiny_loaders[0]))
    out = m(b)

    def grad(**kw):
        loss, st = m.compute_loss(out, b, beta0=0.0, **kw)
        return torch.autograd.grad(loss, out["lv_post"], retain_graph=True)[0], st["kl_z"]
    g_recon, kz = grad(beta=0.0)
    g_elbo, _ = grad(beta=1.0)
    assert not torch.allclose(g_recon, g_elbo)
    assert torch.allclose(grad(beta=1.0, free_bits_z=10 * kz)[0], g_recon)
    assert torch.allclose(grad(beta=1.0, free_bits_z=0.1 * kz)[0], g_elbo)


def test_kl_warmup_schedule():
    from xvgae.train import standard_config, finalise_budget, kl_weights
    cfg = finalise_budget(standard_config("x"), 80)
    assert cfg["epochs"] == 938 and cfg["warmup_epochs"] + cfg["anneal_epochs"] == 47
    assert kl_weights(0, cfg) == (1e-4, 0.1 * 1e-4) and kl_weights(cfg["epochs"] - 1, cfg) == (1.0, 0.1)
    short = finalise_budget(standard_config("x"), 32, updates=6400)
    assert short["warmup_epochs"] + short["anneal_epochs"] == 100
