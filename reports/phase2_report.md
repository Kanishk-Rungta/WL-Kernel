# Self-Calibrated Degree-Adaptive Sliced Wasserstein Weisfeiler–Lehman Kernel: Evaluating Structural Calibration and Computational Efficiency

## 1. Abstract
We evaluate a prespecified structural heuristic that removes global temperature tuning from degree-adaptive SWWL. In the matched paper-grid experiment, SCDA accuracy was 66.00% (sample SD 4.17), versus 64.67% original and 66.33% tuned adaptive. Its wall time was 50.64s versus 210.49s tuned adaptive. These are exploratory findings from one dataset, not proof of better generalization.

## 2. Research background
SWWL combines continuous WL refinement, shared unit projections, empirical quantiles and a distance-substitution kernel for precomputed-kernel SVM classification. The original paper is Carpintero Perez et al., AISTATS 2024, PMLR 238:1297–1305. Phase 1 independently implemented the formula-based pipeline and found a small uncertain adaptive gain accompanied by extra tuning cost.

## 3. Limitations of tuned adaptive SWWL
Nested search over tau multiplies eligible feature configurations and SVM fits. Uniform averaging can suppress structural distinctions; a fixed global tau cannot adapt to graph degree-gap scales. Neither statement guarantees that structural calibration improves predictive accuracy. Phase 2 follows inspection of Phase 1 outcomes and is exploratory.

## 4. Calibration motivation
A graph-local degree-gap summary provides a deterministic scale without labels, cross-graph fitted statistics, or learned attention. The additive constant 1 and median statistic were specified before Phase 2 outer evaluations and were not adjusted after observing results.

## 5. Mathematical formulation
For each unique undirected edge, delta(u,v)=|degree(u)-degree(v)|. Set tau_G=1+median(delta), or 1 for an edgeless graph. Row weights are softmax(-delta(v,u)/tau_G). The update is h_next(v)=0.5*(h(v)+sum_u a(v,u)h(u)), with isolates retaining h(v). Features concatenate h^(0)..h^(H), then use P=Q=20 projected quantiles scaled by 1/sqrt(PQ). Formula style uses linear quantiles and exp(-gamma*||phi_i-phi_j||^2). Augmentation appends raw log1p(degree), without learned scaling. No tau candidates exist for SCDA.

## 6. Implementation methodology
The Phase 1 refinement, Gaussian kernel and experiment functions remain intact. Phase 2 uses separate optimized modules. Degrees and degree gaps are computed once per graph per method; each graph-temperature operator is constructed once and reused across H. Histories are refined to max H once, with prefixes reused. Projection directions are independently constructed for each concatenated dimension using the unchanged NumPy seed convention. Candidate enumeration and first-candidate tie handling match Phase 1. Maps/distances and sparse operators are stored with content-addressed keys including graph-order fingerprint, method, feature mode, H, temperature/calibration, seed, P/Q, style, interpolation, distance, precision and implementation version. Gamma kernels are reused in memory across C values. Cold runs do not read numerical caches; warm mode reuses maps/distances, while structures and histories are deliberately reconstructed. No graph labels enter the cache fingerprint or calibration.

## 7. Dataset
Official ENZYMES: 600 graphs, six classes of 100, 18 continuous attributes, averages 32.633 nodes and 62.137 undirected edges. Categorical node labels are ignored. Archive SHA256 is 13d832eb6ffa084192daf6e5750250028a18437ee692c38d29a10cd60e18aaf4. The archive checksum, historical index-to-label sequence, and regenerated splits were validated before loading the exact saved Phase 1 fold indices. Ordered graph-content fingerprints are recorded for Phase 2 cache keys.

## 8. Experimental setup
Quick uses stratified 5 outer / 3 inner folds: H=[1,2,3], gamma=[0.001,0.01,0.1], C=[0.1,1,10], tuned tau=[0.5,2,5]. Paper-grid uses 10 outer / 5 inner folds: H=[0,1,2,3], gamma=[0.0001,0.001,0.01,0.1], C=10**[-3..3], tuned tau=[0.5,1,2,5]. Seed 42; shared projections for equal dimensions; same SVC, scoring, grids and exact splits. Inner mean accuracy alone chooses H/gamma/C/tau. All five methods are rerun under cold timing in the same environment, sequentially with one numerical-library thread. No outer results are used to select model settings. Additional-seed methods were prespecified as original and SCDA.

## 9. Accuracy, F1 and measured evaluation costs
Accuracy is percent, F1 is on [0,1], times are seconds and memory is MiB. Accuracy SD is sample SD across outer folds. Weighted-F1 equals macro-F1 here because each outer fold is class-balanced.

### Quick formula

| method | accuracy_mean | accuracy_std | macro_f1_mean | weighted_f1_mean | embedding_seconds | search_seconds | total_seconds | svm_fits | peak_rss_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 65.5000 | 5.1908 | 0.6543 | 0.6543 | 0.9844 | 2.5714 | 5.3689 | 410 | 220.6406 |
| adaptive | 66.1667 | 4.6622 | 0.6613 | 0.6613 | 4.5097 | 7.8342 | 19.4929 | 1220 | 295.4062 |
| augmented | 66.3333 | 6.8109 | 0.6651 | 0.6651 | 1.1612 | 2.6751 | 5.8905 | 410 | 226.2891 |
| self_calibrated | 66.6667 | 4.8233 | 0.6654 | 0.6654 | 1.2774 | 2.5148 | 5.6374 | 410 | 226.2305 |
| self_calibrated_augmented | 67.3333 | 7.0317 | 0.6733 | 0.6733 | 2.0111 | 2.6782 | 7.7373 | 410 | 225.2539 |

| total_speedup | search_speedup | feature_speedup | search_seconds_saved | search_percent_saved | accuracy_difference_pp |
| --- | --- | --- | --- | --- | --- |
| 3.4578 | 3.1152 | 3.5304 | 5.3194 | 67.8993 | 0.5000 |

### Paper-grid formula

| method | accuracy_mean | accuracy_std | macro_f1_mean | weighted_f1_mean | embedding_seconds | search_seconds | total_seconds | svm_fits | peak_rss_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 64.6667 | 4.0673 | 0.6468 | 0.6468 | 1.1046 | 50.0511 | 52.8137 | 5610 | 241.3789 |
| adaptive | 66.3333 | 5.2587 | 0.6611 | 0.6611 | 4.9572 | 197.6969 | 210.4855 | 22410 | 425.1328 |
| augmented | 66.0000 | 5.9421 | 0.6607 | 0.6607 | 1.0103 | 53.5516 | 56.3573 | 5610 | 249.7812 |
| self_calibrated | 66.0000 | 4.1722 | 0.6596 | 0.6596 | 1.2217 | 47.5469 | 50.6450 | 5610 | 252.6367 |
| self_calibrated_augmented | 66.3333 | 5.0185 | 0.6608 | 0.6608 | 1.1858 | 52.8771 | 55.9172 | 5610 | 253.6562 |

| total_speedup | search_speedup | feature_speedup | search_seconds_saved | search_percent_saved | accuracy_difference_pp |
| --- | --- | --- | --- | --- | --- |
| 4.1561 | 4.1579 | 4.0576 | 150.1500 | 75.9496 | -0.3333 |

### Quick authors-style

| method | accuracy_mean | accuracy_std | macro_f1_mean | weighted_f1_mean | embedding_seconds | search_seconds | total_seconds | svm_fits | peak_rss_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 66.6667 | 5.4006 | 0.6656 | 0.6656 | 0.9539 | 3.0047 | 5.9746 | 410 | 226.1719 |
| self_calibrated | 67.0000 | 5.7312 | 0.6691 | 0.6691 | 1.3559 | 2.6457 | 6.0826 | 410 | 225.3555 |

### Additional-seed 43 quick formula

| method | accuracy_mean | accuracy_std | macro_f1_mean | weighted_f1_mean | embedding_seconds | search_seconds | total_seconds | svm_fits | peak_rss_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 66.1667 | 4.7361 | 0.6605 | 0.6605 | 0.9836 | 2.1441 | 5.1188 | 410 | 220.4102 |
| self_calibrated | 66.1667 | 4.3141 | 0.6595 | 0.6595 | 1.2124 | 2.7617 | 5.8292 | 410 | 221.6562 |

## 10. Runtime, memory and profiling
Separate preprocessing, operator, WL, projection/quantile, distance, kernel, cache IO, and SVM search/refit durations are in timing.csv. Total is actual per-method wall time including cache writes and fold checkpoint writes; dataset loading and final plots/reports are excluded. compute_seconds excludes cache IO. Compare Phase 2 methods to each other; historical Phase 1 totals excluded IO and should not be used for speedup claims. Absolute RSS is sampled every 10ms, can miss short peaks, includes interpreter/data, and is influenced by sequential allocator retention. Each method begins with independent structures/operators/histories, though operating-system file caches and system load are uncontrolled.

SCDA eliminates temperature search: 5610 fits versus 22410 tuned adaptive. Search time saved was 150.15s (75.9%). Total wall speedup was 4.16x; aggregate eligible feature time speedup was 4.06x. Aggregate feature savings largely reflect fewer tau configurations and must not be described as faster per-graph asymptotic refinement.

Equal-work profiling (three alternating-order repetitions, all 600 graphs, H=0..3, fixed tau=2, no cache IO or SVM search) measured a legacy/optimized mean ratio of 2.30x. This is a constant-factor implementation gain, not a changed asymptotic WL complexity.

| engine | mean | std |
| --- | --- | --- |
| legacy | 2.3128 | 0.2110 |
| optimized | 1.0067 | 0.0871 |

## 11. Temperature distribution
| minimum | maximum | mean | median | std | unique | proportion_equal_one |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 3.0000 | 2.0017 | 2.0000 | 0.2845 | 5 | 0.0383 |

| temperature | graphs |
| --- | --- |
| 1.0000 | 23 |
| 1.5000 | 4 |
| 2.0000 | 544 |
| 2.5000 | 6 |
| 3.0000 | 23 |

Temperatures do vary, but variation is limited around the median of 2. A graph-specific formula alone is not evidence that graph specificity caused any gain; a fixed-tau=2 comparison would be needed to isolate that question. The rule was left unchanged. Edge-gap summary (each undirected edge once):

| statistic | degree_gap |
| --- | --- |
| count | 37282.0000 |
| mean | 1.0639 |
| std | 0.9930 |
| min | 0.0000 |
| 25% | 0.0000 |
| 50% | 1.0000 |
| 75% | 2.0000 |
| max | 7.0000 |

Historical inner-selected globally tuned tau values, descriptive only:

| globally_selected_tau | outer_folds |
| --- | --- |
| 0.5000 | 4 |
| 1.0000 | 4 |
| 5.0000 | 2 |

The global tau is selected for classification using training labels; tau_G is a label-free graph statistic. Their distributions are not estimates of the same quantity.

## 12. Ablation results
Original versus augmented isolates the degree attribute subject to a projection-dimension change. Original versus SCDA changes structural aggregation. SCDA versus SCDA+augmented adds the degree attribute; tuned adaptive versus SCDA changes temperature choice and search flexibility. The original combined model remains available, with all Phase 1 outputs preserved, but is not required in this five-method primary comparison. In the paper-grid run, augmented accuracy was 66.00% and SCDA+augmented was 66.33%. These are descriptive ablations, not evidence of causal mechanism.

## 13. Paired statistical analysis
| method | reference | metric | mean_difference | std_difference | descriptive_ci_low | descriptive_ci_high | exploratory_t_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| self_calibrated | original | accuracy | 1.3333 | 2.1943 | -0.2364 | 2.9030 | 0.0868 |
| self_calibrated | adaptive | accuracy | -0.3333 | 2.1943 | -1.9030 | 1.2364 | 0.6424 |
| self_calibrated | augmented | accuracy | 0.0000 | 4.2310 | -3.0267 | 3.0267 | 1.0000 |

Fold-level changes for accuracy, macro-F1 and weighted-F1 are saved. Intervals are descriptive 95% Student-t intervals and tests are exploratory paired t tests. Overlapping training sets make folds dependent; sample size is small and comparisons are multiple. No independence-adjusted confirmatory inference or statistical-significance claim is justified. The prespecified additional seed 43 quick check gave baseline 66.17% and SCDA 66.17%, difference +0.00 pp. It is reported irrespective of direction and was not used to select a seed or change calibration.

## 14. Authors-style reference comparison
Matched quick authors-style accuracy: baseline 66.67%, SCDA 67.00%; difference +0.33 pp. These are separate conventions; do not attribute differences between styles to calibration. A full authors-style paper-grid search was not run.

Actual official source intermediate-check status: passed; maximum quantile difference: 2.220446049250313e-16; maximum kernel difference: 0.0. See authors_style_reference.md for source hashes, nugget convention, precision and RNG limitations. Authors-style means matched quantile/kernel conventions, not an exact official experimental reproduction.

## 15. Accuracy–efficiency trade-offs
SCDA changed accuracy by +1.33 pp versus original and -0.33 pp versus tuned adaptive. Figures/accuracy_runtime.png highlights the observed Pareto frontier: no other displayed method has at least as high accuracy and at most as much wall time, with one strict inequality. This ignores uncertainty and is not a rule for selecting a model on outer tests. Reduced search cost is directly measurable; structural weighting mechanisms and stronger generalization remain hypotheses. Retaining accuracy descriptively does not demonstrate statistical noninferiority or equivalence.

In this paper-grid run, SCDA and SCDA+augmented form the observed frontier. SCDA+augmented matched the tuned adaptive mean accuracy with 3.76x less wall time; SCDA was the faster of the two calibrated variants with slightly lower mean accuracy. This is a measured trade-off, not a single universally best model.

## 16. Limitations and threats to validity
One small dataset, mostly one seed, exploratory follow-up to earlier results, dependent folds, additional tuning flexibility, coarse temperature variation, projection changes with attribute dimension, sequential timing, cache IO cost, absolute sampled memory, and independent NumPy rather than official Torch RNG/precision. An additional quick seed is limited robustness evidence. No constants or grids were changed after inspecting Phase 2 held-out outcomes. The published 66.67% +/-5.00% is external reference, not our measurement, and should not support a superiority claim.

## 17. Conclusions
The strongest defensible contribution is a deterministic structural calibration heuristic that removes tau model selection, together with measured, matched nested-CV accuracy–cost comparisons and reproducible implementation optimizations. Observed accuracy differences remain uncertain. Any negative or inconsistent seed/style outcomes are reported alongside positive ones. Fewer hyperparameters do not automatically guarantee better generalization.

## 18. Future research
Prespecified fixed-tau controls, more datasets and seeds, exact Torch environment replication, controlled hardware profiling and appropriate repeated-CV inference could test whether graph-specific calibration contributes beyond eliminating search. Alternative calibration equations would be new hypotheses requiring fresh validation.

## 19. References and reproducibility
Carpintero Perez, R., Da Veiga, S., Garnier, J., Staber, B. (2024), Gaussian process regression with Sliced Wasserstein Weisfeiler–Lehman graph kernels. https://proceedings.mlr.press/v238/carpintero-perez24a.html . Official source https://gitlab.com/drti/swwl, inspected commit a974d59ecb208758844ef15381175cb8b1333c05. Dataset https://chrsmrrs.github.io/datasets/docs/datasets/ . Exact commands in README.md. Phase 1 file hashes are stored in results/phase2/logs/phase1_manifest.json and verified separately. Nothing is pushed to GitHub during Phase 2.

All 44 tests passed after integration. The artifact audit verified 127 preserved Phase 1 files and each research run's 600 unique out-of-fold predictions per method, fold summaries and inner-selected scores. Primary paper-grid figures are in results/phase2/paper/figures: accuracy.png, macro_f1.png, runtime.png, accuracy_runtime.png, temperature_distribution.png, speedup.png, paired_differences.png, and edge_degree_gaps.png. Matched authors-style figures are in results/phase2/authors_style/figures. All are 300dpi PNGs with captions.md explanations.
