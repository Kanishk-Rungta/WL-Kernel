# Degree-Adaptive SWWL on ENZYMES

Independent reproduction of the **Original SWWL (AISTATS 2024 reproduction)** paper formulas and three prespecified ablations: adaptive aggregation, log-degree augmentation, and their combination. Real experiments use raw continuous attributes, normalized projected quantiles, and precomputed-kernel SVMs. CPU only.

## Installation
Python 3.10+ is required. From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py --download-only
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python main.py --mode smoke
.\.venv\Scripts\python main.py --mode quick --seed 42
.\.venv\Scripts\python main.py --mode paper --seed 42
.\.venv\Scripts\python main.py --mode quick --figures-only
```

On Linux/macOS replace `.\.venv\Scripts\python` with `.venv/bin/python`. Alternatively activate the environment and use `python`. The original experiment environment inherited existing system packages; use the fresh environment commands above for this checkout. requirements-lock.txt records the session environment; logs/config.json identifies the actual relevant package versions.

```text
python main.py --help
python main.py --mode quick --methods original adaptive augmented combined
```

The official dataset downloads automatically from https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip and is checksum-recorded. Labels remain 1 through 6. Categorical node labels are ignored.

## Evaluation and outputs
Quick uses stratified 5-fold outer / 3-fold inner CV, H=[1,2,3], gamma=[0.001,0.01,0.1], C=[0.1,1,10], tau=[0.5,2,5], P=Q=20. Paper uses 10/5 folds, H=[0,1,2,3], gamma=[0.0001,0.001,0.01,0.1], C=10**[-3,...,3], tau=[0.5,1,2,5]. Smoke uses 48 graphs and is debugging only. Default seed is 42. Inner accuracy selects hyperparameters; outer folds are never used for tuning. Maps use no learned preprocessing and are safe to precompute on all graphs. Uniform/infinite-temperature refinement agrees exactly. Isolates retain their features. The appended feature is raw log1p(degree), without dataset normalization.

Quick results: `results/tables`, `results/figures`, `results/logs`, `results/cache`; other modes use `results/smoke` or `results/paper` so quick evidence is preserved. Rerunning a mode overwrites that mode's outputs. comparison.csv includes means and sample standard deviations, timing, absolute RSS and deltas. Fold-level results, parameters, complete validation scores, OOF predictions, paired descriptive statistics, and splits are saved. Cached NPZ files contain maps and squared distances; cold extraction is deliberately recomputed for comparable timing. See figures/captions.md and reports/experiment_report.md.

## Scientific scope
Paper and code differ: official SWEncoder uses lower quantiles and the classification script uses exp(-gamma*EuclideanDistance); here linear interpolation and exp(-gamma*SquaredDistance) follow the paper formulas and requested specification. NumPy projections differ from Torch projections. Reference source is inspected from the official repository, commit a974d59ecb208758844ef15381175cb8b1333c05. No authors' code was copied. This is not an exact official-code replication. The published 66.67% +/- 5.00% is a reference, not our measurement. Quick results cannot be directly compared to the published nested 10-fold result. Repeated seeds and additional datasets are needed for strong conclusions. Sampled absolute process RSS includes retained allocations and is not method-specific incremental memory.

Sources: https://proceedings.mlr.press/v238/carpintero-perez24a.html ; https://gitlab.com/drti/swwl ; https://chrsmrrs.github.io/datasets/docs/datasets/ .

## Recorded results

Both quick and paper-grid evaluations completed locally with seed 42; 18 tests passed.

| Method | Quick accuracy (%) | Paper-grid accuracy (%) | Paper-grid macro-F1 |
|---|---:|---:|---:|
| Original | 65.50 | 64.67 | 0.6468 |
| Adaptive | 66.17 | 66.33 | 0.6611 |
| Augmented | 66.33 | 66.00 | 0.6607 |
| Combined | 66.50 | 66.17 | 0.6598 |

All descriptive paired accuracy intervals span zero; no reliable improvement is established. See [quick report](reports/experiment_report.md), [paper-grid report](reports/paper_report.md), and [reference review](reports/reference_review.md). Dataset downloads, Python environments and regenerable numeric caches are excluded from Git. All measured tables, configurations, splits, predictions and figures are included.
