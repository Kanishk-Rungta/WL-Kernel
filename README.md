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


## Phase 2: self-calibrated degree-adaptive SWWL

SCDA-SWWL replaces the globally tuned temperature with `tau_G = 1 + median(abs(degree(u)-degree(v)))`, counting each undirected edge once; edgeless graphs use 1. Weights are row softmaxes of negative degree gap divided by tau_G. The constant and calibration rule are prespecified heuristics, not learned or adjusted from held-out results. Both `self_calibrated` and `self_calibrated_augmented` are available; the latter appends raw log1p(degree).

Phase 1 functions and outputs are preserved. Phase 2 uses separate optimized operators and incremental WL histories, with separate projections for every H dimension. Label-independent content-addressed caches distinguish dataset order, method, features, H, temperature/calibration, projection seed/P/Q, kernel style, interpolation, distance, precision and implementation version. Gamma kernels are held in memory across C candidates. Default cold runs recompute features; `--cache-mode warm` reuses cached maps/distances. Structures/operators/histories are still reconstructed in warm mode. Graph operators are also saved for inspection and explicit reuse via the cache API. Phase 2 methods all follow the same documented timing protocol.

No new dependencies are required. From this project directory using the existing workspace environment:

```powershell
..\.venv\Scripts\python -m pip install -r requirements.txt
..\.venv\Scripts\python -m pytest -q
..\.venv\Scripts\python main.py --mode smoke --phase 2
..\.venv\Scripts\python main.py --mode quick --phase 2 --seed 42
..\.venv\Scripts\python main.py --mode paper --phase 2 --seed 42
..\.venv\Scripts\python main.py --mode quick --phase 2 --kernel-style authors
..\.venv\Scripts\python main.py --mode quick --phase 2 --seed 43 --methods original self_calibrated
..\.venv\Scripts\python main.py --mode quick --phase 2 --figures-only
..\.venv\Scripts\python main.py --mode paper --phase 2 --figures-only
..\.venv\Scripts\python main.py --mode quick --phase 2 --kernel-style authors --figures-only
..\.venv\Scripts\python -m src.phase2_profiling
..\.venv\Scripts\python -m src.phase2_reporting
```

For a fresh checkout create `.venv` and use `.\.venv\Scripts\python` instead. On Linux/macOS use `.venv/bin/python`. Exact session versions are in requirements-lock.txt and each run's logs/config.json. Optional `python -m src.phase2_reference` verifies intermediate maps against a sibling `reference_swwl` checkout and an already available Torch installation; experiments never require Torch.

Original commands retain their four-method Phase 1 defaults. Specifying `--phase 2`, `--kernel-style authors`, or a new self-calibrated method routes outputs to Phase 2. Default Phase 2 formula methods are original, adaptive, augmented, self_calibrated, self_calibrated_augmented. Authors-style defaults are original and self_calibrated, matched within that style. `combined` remains available explicitly. `--run-id NAME` creates a named output subdirectory. Existing Phase 2 runs are preserved by placing reruns in a timestamped subdirectory; use the printed path and matching `--run-id` to regenerate that rerun's figures. Phase 1 commands retain their historical overwrite behavior; use Phase 2 commands for new research runs.

Outputs are in `results/phase2/quick`, `paper`, `authors_style`, `profiling`, and `cache`. Seed 43 is saved in `quick/seed43`. Reports are `reports/phase2_report.md` and `reports/authors_style_reference.md`. Formula and authors-style outputs never share directories or incompatible cache entries. The authors-style option uses lower quantiles, unsquared distance and the inspected 1e-7 training nugget; its NumPy RNG and float64 precision remain independent of official Torch defaults.

Quick/full grids, SVC scoring and seed-42 saved outer/inner folds match Phase 1 exactly after validating archive checksum and index-to-label order. Only inner accuracy selects model parameters. SCDA has no tau grid; tuned adaptive retains its original grid and first-tie decisions. This investigation is exploratory because it follows Phase 1 outcomes. Speedup from eliminating tau candidates is reduced model-selection cost, not a new asymptotic refinement complexity. Per-graph structural calibration may have limited variation; this limitation is measured rather than hidden. A paired interval containing zero does not establish an improvement or noninferiority. Pareto plots describe measured means/runtimes and ignore uncertainty; they are not outer-test model-selection rules.

### Phase 2 measured findings

Full matched 10x5 formula evaluation: original 64.67% +/-4.07, tuned adaptive 66.33% +/-5.26, augmented 66.00% +/-5.94, SCDA 66.00% +/-4.17, SCDA+augmented 66.33% +/-5.02. SCDA used 50.64s wall time and 5,610 SVM fits versus 210.49s and 22,410 for tuned adaptive. Search saved 150.15s (75.95%); total speedup was 4.16x. This is primarily reduced model-selection cost. SCDA's paired accuracy changes were +1.33 pp versus original and -0.33 pp versus tuned adaptive; both descriptive 95% intervals contain zero. No reliable accuracy gain or noninferiority is established.

SCDA and SCDA+augmented form the observed paper-grid accuracy/runtime frontier. SCDA+augmented matched tuned adaptive mean accuracy with 3.76x less wall time. The frontier ignores uncertainty and does not choose a model using outer tests. Equal-work fixed-tau feature profiling measured about 2.30x legacy/optimized mean speedup across three alternating-order repetitions, reflecting implementation reuse rather than changed WL asymptotic complexity.

Temperatures: minimum 1, maximum 3, mean 2.0017, median 2, sample SD 0.2845; 544/600 graphs use exactly 2, and 23/600 use 1. Calibration has limited graph-to-graph variation. Authors-style quick accuracy was 66.67% original versus 67.00% SCDA; seed-43 formula quick was a tie at 66.17%. These findings limit claims of a robust accuracy benefit. Full authors-style paper-grid and broad repeated-seed/multi-dataset validation remain future work.

All 44 tests passed. All 127 tracked Phase 1 files retained their recorded hashes. Research-run artifact audits verified 600 unique out-of-fold predictions per method, matching fold summaries and inner-selected scores. See [Phase 2 report](reports/phase2_report.md) and [authors-style reference](reports/authors_style_reference.md). No Phase 2 GitHub push was performed.
