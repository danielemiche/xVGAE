# xVGAE: a hierarchical variational graph autoencoder for exchangeable graphs

Code for **xVGAE** (Micheletti, Ricci, Sudderth; NeurIPS 2026; [arXiv:2610.30815](https://arxiv.org/abs/2610.30815)). A graph with n nodes is generated as
z0 ~ p(z0 | n) (a DDPM), z_i ~ p(z_i | z0) for every node, A_ij ~ Bernoulli(σ(s(z_i, z_j, z0))).

```bash
# 1. install (Python 3.10; for a GPU use --index-url https://download.pytorch.org/whl/cu128)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e . --no-deps && bash scripts/build_orca.sh
# 2. data: a paper corpus (its edges are checked against the paper's), sbm2, or your own graphs (custom)
python scripts/make_data.py mmsbm_medium --out data
# 3. stage 1, its checkpoint picked on validation, then the z0 DDPM (GPU: 2-3 h and under 1 h; the steps after stage 1 also run on a CPU)
python scripts/train_stage1.py --config configs/mmsbm_mlp.json --data data --run-dir runs/mmsbm_mlp
python scripts/select_stage1.py --run runs/mmsbm_mlp
python scripts/train_z0_ddpm.py --run runs/mmsbm_mlp --out-dir runs/mmsbm_mlp/z0_ddpm
# 4. 3 test draws (the DDPM state is first picked on validation), then the scores
python scripts/sample.py --run runs/mmsbm_mlp --ddpm-dir runs/mmsbm_mlp/z0_ddpm --out-dir samples/mmsbm_mlp
python scripts/score.py --name mmsbm_medium --data data --samples samples/mmsbm_mlp
```

Without `--config`, `train_stage1.py --data data --name <corpus> --decoder sab|mlp` trains the standard recipe.
The model is in `xvgae/` (`encoder.py`, `decoder.py`, `prior.py`, `model.py`, `diffusion.py`; `train.py`, `metrics.py`,
`data.py`, `corpora.py`). Data and your own graphs: [docs/DATA.md](docs/DATA.md). Recipe, paper rows and CPU runs:
[docs/TRAINING.md](docs/TRAINING.md). Scores: [docs/METRICS.md](docs/METRICS.md). Tests: `python -m pytest tests -q`.

Code: MIT. ORCA (GPL-3.0) and the CiteSeer / D&D data are downloaded, not redistributed. Citation: `CITATION.cff`.
