# Data

A corpus is `<root>/<name>/<name>_{train,val,test}.pt`, each a `torch.save`d list of `torch_geometric.data.Data`
(`<root>` = `--data`, else `$XVGAE_DATA_DIR`, else `./data`):

| field | shape | content |
|---|---|---|
| `x`, `eigvecs` | [n, k] float32 | the k smallest eigenvectors of the normalised Laplacian (largest \|entry\| of each column positive, zero columns when n < k); the encoder reads both |
| `eigvals` | [1, k] float32 | the matching eigenvalues |
| `edge_index` | [2, 2m] int64 | both directions of every edge, no self-loops |
| `num_nodes` | int | n |

Only the encoder sees `x`; the scores use only the edges.

## Your own graphs

```bash
python scripts/make_data.py custom --graphs my_graphs.pkl --name mine --out data     # then --data data --name mine
python scripts/make_data.py sbm2 --out data                                          # example: 400 two-block SBM graphs, n 40-120
```

`--graphs` is a `.pkl` or `.pt` file with a list of `networkx` graphs or PyG `Data` (`edge_index`, `num_nodes`), or a
dict `{"train": [...], "val": [...], "test": [...]}` to keep your split (otherwise a seeded 64 / 16 / 20 % split).
`--k` sets the eigenvectors per node (default: the smallest n, between 8 and 50); `--lcc` keeps each graph's largest
component. The script also writes `floor_selection.json`, which the trainer uses to pick `model_best_gen.pt`.
From Python: `xvgae.data.build(graphs, "mine", out="data")`. Graphs up to a few hundred nodes are the intended range
(the decoder scores all n(n-1)/2 pairs).

## The paper corpora

| key | graphs (train / val / test) | n | source | build time |
|---|---|---|---|---|
| `citeseer_ego` | 484 / 121 / 152 | 50-399 | 3-hop ego graphs of CiteSeer (Sen et al. 2008), downloaded | ~1 min |
| `mmsbm_medium` | 640 / 160 / 200 | 120-400 | mixed-membership SBM, K in {3, 5}, seeded | 2-3 min |
| `lfr_v2` | 640 / 160 / 200 | 200-500 | LFR benchmark, every parameter drawn per graph, seeded | ~20 min |
| `protein` | 587 / 147 / 184 | 100-500 | D&D protein graphs (Dobson & Doig 2003), downloaded | a few min |

Every split is checked against the paper corpus edges (SHA-256 in `xvgae/paper_reference.json`) when it is loaded.
CiteSeer and D&D are downloaded through PyTorch Geometric; cite Sen et al. (2008), Yang et al. (2016), Dobson & Doig
(2003) and Morris et al. (2020) when you use them.
