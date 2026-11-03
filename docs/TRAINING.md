# Training

## The standard recipe

`python scripts/train_stage1.py --data data --name <corpus> --decoder sab|mlp` (defaults in `xvgae/model.py` and
`xvgae/train.py`):

| part | setting |
|---|---|
| latents and network | z dim 8, z0 dim 32, hidden 64, 5 layers; decoder with 8 attention and 8 bilinear heads (rank 4) |
| decoder | `sab`: 5 set-attention layers across nodes (xVGAE-SAB); `mlp`: pointwise (xVGAE-MLP) |
| priors | p(z_i \| z0) and p(z0 \| n) Gaussian, both N(0, I) at initialisation |
| loss | negative ELBO; β = 1, β0 = 0.1; KL warm-up over 5 % of the budget (at least 3750 updates): a quarter at 1e-4, then a half-cosine ramp |
| optimiser | AdamW, lr 5e-4 cosine-decayed to 2.5e-5, weight decay 1e-5, batch 8, gradient clip 1, random eigenvector sign flips |
| budget | 75k updates or 10 h (`--updates`, `--max-walltime-h`) |
| z0 DDPM | 20 posterior z0 draws per training graph, 2000 epochs, lr 2e-4, batch 64; states at epochs 1000 and 2000 and the best held-out loss, one picked on validation by `sample.py` |

Checkpoints: `model_last.pt`, `model_best_gen.pt` (best validation MMD of encoded z0 with z from the prior during
training, as the geometric mean of the ratios to the selection floor), `model_best_warmup.pt` (lowest validation loss
during the KL warm-up) and 20 snapshots `model_ep<E>.pt` (`--snapshots`). `resume.pt` is written every 15 minutes: a
stopped run continues with `--resume <run dir>`.

## Picking the checkpoint and the DDPM state

Both picks use the validation split only; the test split is drawn once, at the end.

1. `scripts/select_stage1.py --run <run>` screens every `model_*.pt`. Each draws 3 validation sets (seeds 0-2): z0 from
   the encoder, one draw of q(z0 | G) per validation graph at that graph's n, and the node latents from p(z | z0). The
   pick goes to `<run>/stage1_pick.json`, which `train_z0_ddpm.py` and `sample.py` read by default (`--ckpt` names another).
2. `scripts/sample.py` picks the DDPM state the first time it runs: 10 validation sets per state (seeds 0-9), z0 from the
   DDPM and n from the training sizes. The pick goes to `<ddpm-dir>/state_pick.json`. The stage 1 is always the one the
   DDPM was trained on (`ddpm_z0_meta.json`).

The rule (`xvgae.metrics.Selector`): each set is scored against the validation split, with MMD^2 under the EMD kernel
(bandwidths fitted on the training split) for degree, clustering, spectral, orbit and WL, and the C2ST accuracy. Both are
averaged over the sets. The lowest geometric mean of the five MMD^2 holds the pick. A candidate within 10 % of it takes the
pick only when its C2ST is closer to 0.5 beyond noise: |acc - 0.5|, set to 0 within 2 null SDs of 0.5, lower by more than
2 SE. A checkpoint is a candidate only when it passes the validity gate itself on the validation split (edge ROC AUC
>= 0.65, kl_z >= 0.05). `--skip-warmup` leaves out the checkpoints from inside the KL warm-up. Both picks keep each
candidate's score (`stage1_scores/`, `state_scores/`), so a stopped pick continues; `--draws` and `--val-draws` set the
number of validation sets.

## Validity gate

`VALIDITY.json` and the exit code of `train_stage1.py`; do not sample from or report a run that is not valid.

| exit | verdict | next |
|---|---|---|
| 0 | valid | pick the checkpoint, then sample |
| 10 | collapse: kl_z < 0.05 nats per node and dimension, or validation ROC < 0.65 | retrain with `--free-bits 0.15` and twice the `--updates` (with `--config`: twice the `epochs` in a copy of the config) |
| 11 | undertrained: validation loss still falling by > 1 % over the last 10 % of the epochs | retrain with more `--updates` |
| 12 | stopped by the wall-time cap | `--resume <run dir>` |
| 13 | no training trajectory in the run directory | train again |

Other levers of the standard recipe: `--beta0` (the Protein MLP row uses 1), `--decoder mlp`. With `--config`, the config
fixes the decoder, β0 and the epochs: change them in a copy of the config.

## The paper rows

One config per row of the paper's table, `configs/<row>.json`: `citeseer_ego_sab`, `citeseer_ego_mlp`, `mmsbm_sab`,
`mmsbm_mlp`, `lfr_sab`, `lfr_mlp`, `protein_sab`, `protein_mlp`. Each row runs the steps of the README:

```bash
python scripts/train_stage1.py --config configs/<row>.json --data data --run-dir runs/<row>
python scripts/select_stage1.py --run runs/<row>
python scripts/train_z0_ddpm.py --run runs/<row> --out-dir runs/<row>/z0_ddpm
python scripts/sample.py --run runs/<row> --ddpm-dir runs/<row>/z0_ddpm --out-dir samples/<row>
python scripts/score.py --name <corpus> --data data --samples samples/<row>
```

## Short CPU runs and cost

On 4 CPU threads one update takes 0.2-0.5 s at n ~ 80, so the full budget needs a GPU. A quick model:

```bash
python scripts/make_data.py sbm2 --out data
python scripts/train_stage1.py --data data --name sbm2 --updates 6400 --snapshots 0 --run-dir runs/sbm2
python scripts/select_stage1.py --run runs/sbm2      # --force if this short budget does not pass the validity gate
python scripts/train_z0_ddpm.py --run runs/sbm2 --out-dir runs/sbm2/z0_ddpm --epochs 300
python scripts/sample.py --run runs/sbm2 --ddpm-dir runs/sbm2/z0_ddpm --out-dir samples/sbm2
python scripts/score.py --name sbm2 --data data --samples samples/sbm2
```

About 35 min on 4 threads (peak 0.8 GB). A quick model, not the paper recipe.

Approximate cost with the configs in `configs/`:

| step | one RTX 6000 Ada | 8 CPU threads |
|---|---|---|
| stage 1, 938-1230 epochs | 1.7-2.7 h | - |
| `select_stage1.py`, 23 checkpoints | - | 7 min-2.1 h (MMSBM) |
| `train_z0_ddpm.py`, 2000 epochs | under 1 h | 47-84 min |
| `sample.py`: DDPM state pick, 3 test draws | - | 14-71 min (MMSBM) |
| `score.py`, 3 draws: TV; EMD | - | 1-10 min; 1.5-19 min |

On 4 CPU threads stage 1 takes 4-20 h, so train it on a GPU. Sampling
and scoring run on the CPU: peak 6.5 GB on MMSBM, at most 2.6 GB on the other corpora.
