# Metrics

`scripts/score.py` (code: `xvgae/metrics.py`). Both the samples and the test split are reduced to their largest
connected components. Per draw, the biased MMD² against the test split of five statistics:

| statistic | per graph | kernel (default, `--kernel emd`) |
|---|---|---|
| degree | one bin per integer degree | Gaussian on the 1-D earth mover's distance (EMD) |
| clustering | 100 bins on [0, 1] | Gaussian on the EMD |
| spectral | normalised-Laplacian eigenvalues, 100 bins on [0, 2] | Gaussian on the EMD |
| orbit | one histogram per 4-node orbit (ORCA), 15 orbits | Gaussian on the EMD per orbit, MMD² summed over the orbits |
| WL | Weisfeiler-Lehman subtree features | linear |

Each bandwidth is the median EMD between pairs of training graphs, so no kernel saturates and all five statistics
enter the aggregate. The same kernels pick the checkpoints on validation ([docs/TRAINING.md](TRAINING.md)).
`--saturation report` prints, per statistic, 1 - the mean kernel value between test graphs (^ marks values ≥ 0.9).

C2ST: a logistic regression on random-GIN features, 5-fold accuracy, reported as C2ST_Δ = |acc - 0.5|.
Ratio = MMD² / floor, where the floor is the mean of the same score for 5 random subsets of |test| training graphs.
A row is the median ratio over 3 draws (seeds 100-102) with its range; Agg is the geometric mean of the ratios over
the ranked statistics.

`--kernel tv` gives the Gaussian-TV kernels of the GraphRNN protocol: σ = 1 for degree and spectrum, σ = 30 on the mean
orbit vector, and for clustering σ = the median TV distance between training graphs (`--sigma-clustering fitted`) or
0.1 (`published`). Under these kernels the orbit statistic can saturate on dense graphs: it is left out of Agg on the
paper corpora where it does, and `--saturation exclude` does the same on any corpus. On a paper corpus
the TV floor is stored (`--measure-floor` measures it); under the EMD kernels the floor is always measured.
