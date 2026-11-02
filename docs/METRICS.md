# Metrics

`scripts/score.py` (code: `xvgae/metrics.py`). Both the samples and the test split are reduced to their largest
connected components. Per draw, the biased MMD² against the test split of five statistics:

| statistic | per graph | kernel (`--kernel tv`, default) |
|---|---|---|
| degree | one bin per integer degree | Gaussian on the TV distance, σ = 1 |
| clustering | 100 bins on [0, 1] | σ = median TV distance between training graphs (`--sigma-clustering fitted`, default) or 0.1 (`published`) |
| spectral | normalised-Laplacian eigenvalues, 100 bins on [0, 2] | σ = 1 |
| orbit | mean 4-node orbit counts (ORCA) | σ = 30 |
| WL | Weisfeiler-Lehman subtree features | linear |

C2ST: a logistic regression on random-GIN features, 5-fold accuracy, reported as C2ST_Δ = |acc - 0.5|.
Ratio = MMD² / floor, where the floor is the mean of the same score for 5 random subsets of |test| training graphs.
A row is the median ratio over 3 draws (seeds 100-102) with its range; Agg is the geometric mean of the ratios over
the ranked statistics (with the TV kernels orbit is not ranked on CiteSeer-Ego, MMSBM and LFR; with `--kernel emd` all five
are, as in the validation selection of docs/TRAINING.md). On a paper corpus the paper's floor and
clustering bandwidth are used (`--measure-floor` measures the floor); on your own corpus both are measured.

The clustering bandwidth is fitted on the training graphs (the paper's protocol; `--sigma-clustering published` uses
0.1). Further options: `--kernel emd` (Gaussian on the 1-D earth mover's distance, bandwidths from the training graphs) and
`--saturation report|exclude` (1 - mean kernel value between test graphs per statistic; ^ marks values ≥ 0.9, and
`exclude` leaves those out of Agg).
