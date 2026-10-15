# Degree-Adaptive Sliced Wasserstein Weisfeiler-Lehman Kernel

## Research problem and source
This project studies graph classification with continuous attributes using a CPU implementation of SWWL. The original contribution is the SWWL pipeline; the proposed contributions here are degree-dependent neighbor weighting and a single structural degree attribute. Their benefit is a hypothesis, not an established fact.

Carpintero Perez, R., Da Veiga, S., Garnier, J., and Staber, B. (2024). Gaussian process regression with Sliced Wasserstein Weisfeiler-Lehman graph kernels. AISTATS, PMLR 238:1297-1305. https://proceedings.mlr.press/v238/carpintero-perez24a.html

## Original methodology and mathematical formulation
Initialize node vectors from the 18 raw continuous attributes. Uniform refinement is h_next(v)=0.5*(h(v)+mean_neighbors(h(u))). Retain all steps 0 through H. Project concatenated vectors onto P shared random unit directions. Evaluate Q evenly spaced quantiles including both endpoints, and flatten with normalization 1/sqrt(P*Q). Squared Euclidean distance between these maps defines K(i,j)=exp(-gamma*d_squared(i,j)). SVC uses the precomputed kernel.

## Limitation and proposed modifications
Uniform aggregation ignores the structural positions of neighbors. Adaptive weighting uses softmax(-abs(degree(v)-degree(u))/tau), computed with row-maximum subtraction. Infinite tau recovers uniform weights. Augmentation appends raw log1p(degree), with no learned scaling. The combined variant uses both. Isolated nodes retain their own attributes at each step.

## Reference inspection and deviations
The official repository was downloaded into ../reference_swwl and its encoders, distances, classification loader and classifier were inspected. Reference commit: a974d59ecb208758844ef15381175cb8b1333c05. This is an independent NumPy/SciPy reproduction, not execution of the authors' environment. The official SWEncoder uses lower interpolation; the paper explicitly describes linear interpolation, which this implementation follows. The official distance function returns Euclidean distance, and its exponential classifier applies that distance directly; our requested baseline follows the squared distance Gaussian in Definitions 1 and Property 1. These discrepancies prevent a claim of exact code reproduction. NumPy PCG64 projections and float64 calculations also differ from Torch random samples and precision. No categorical labels, feature standardization, or supervised preprocessing are used. Paper text contains a dimension typo near Eq.12; implementation uses (H+1)*d as Definition 2 specifies.

## Dataset
ENZYMES from the official TU download: 600 graphs, six classes, 100 per class, 18 attributes, average 32.633 nodes and 62.137 undirected edges. Original class labels 1 through 6 are retained. Archive SHA256 and verified statistics are in logs/config.json. Smoke mode uses a small diagnostic subset and is not research evidence.

## Experimental setup
This run is **quick**. Full configuration and software versions are saved in logs/config.json. Identical stratified outer and inner splits are saved in logs/splits.json. Hyperparameters are selected solely by mean inner accuracy, with first-candidate deterministic ties. Outer tests are evaluated once for the selected configuration. The projection seed is fixed and directions are identical whenever dimensions match. Label-independent graph-local maps can safely be computed before splitting. No parameter was adjusted after examining outer outcomes.

Quick: 5 outer folds, 3 inner folds; H=[1,2,3], gamma=[0.001,0.01,0.1], C=[0.1,1,10], adaptive tau=[0.5,2,5], P=Q=20. Paper mode: 10/5 folds with H=[0,1,2,3], gamma=[0.0001,0.001,0.01,0.1], C=10**[-3,...,3], tau=[0.5,1,2,5]. Paper-compatible describes fold counts and grids, not identical random splits or official-code reproduction.

## Baseline, modified performance, and ablations
The following CSV table gives outer-fold means and sample standard deviations (accuracy in percent; F1 on 0-1 scale), absolute and relative changes from baseline, timings, and sampled process memory:

```csv
method,accuracy_mean,accuracy_std,macro_f1_mean,macro_f1_std,weighted_f1_mean,weighted_f1_std,embedding_seconds,kernel_seconds,search_seconds,total_seconds,peak_rss_mib,svm_fits,candidates_per_fold,accuracy_difference_pp,relative_improvement_percent
original,65.5,5.190803834132476,0.6543105244789968,0.051103548849244,0.6543105244789968,0.0511035488492441,1.6526643999968655,0.1732893000007607,1.865662500000326,3.691616199997952,212.328125,410,27,0.0,0.0
adaptive,66.16666666666666,4.662200243566457,0.6613407611770878,0.0450149266268354,0.6613407611770878,0.0450149266268354,6.30458200002613,0.5339705000078538,5.799705999990692,12.638258500024676,265.91015625,1220,81,0.6666666666666572,1.0178117048345912
augmented,66.33333333333334,6.810938913900849,0.6650997910045982,0.067978111795672,0.6650997910045982,0.067978111795672,1.951477300011902,0.2726157999859424,2.4392640000005485,4.663357099998393,215.96484375,410,27,0.8333333333333428,1.2722646310432717
combined,66.5,6.51920240520265,0.665587674637158,0.0634204662043655,0.6655876746371581,0.0634204662043656,6.77319800000987,0.5601625999988755,5.383638600003906,12.716999200012651,265.46875,1220,81,1.0,1.5267175572519085

```

Measured baseline accuracy is 65.50% +/- 5.19% (sample SD), macro-F1 0.6543. Its numerical difference from the published mean is -1.17 percentage points, without protocol equivalence. adaptive: 66.17% +/- 4.66%, macro-F1 0.6613, accuracy delta +0.67 pp. augmented: 66.33% +/- 6.81%, macro-F1 0.6651, accuracy delta +0.83 pp. combined: 66.50% +/- 6.52%, macro-F1 0.6656, accuracy delta +1.00 pp. adaptive descriptive paired 95% interval: [-1.69, 3.03] pp. augmented descriptive paired 95% interval: [-2.52, 4.19] pp. combined descriptive paired 95% interval: [-2.69, 4.69] pp. The observed changes do not establish a reliable improvement: the descriptive accuracy intervals overlap zero. This run gives no adequate evidence of statistical significance.

The method with highest observed mean accuracy is **combined** (66.50%). This ranking describes these folds and does not select a deployable model using test performance. A/B isolate weighting; A/C examine degree augmentation; D combines both. Different attribute dimensions change the random projection geometry, which is a limitation of the augmentation comparison. A possible benefit of weighting is preservation of similar-degree neighborhoods; a possible harm is suppressing useful dissimilar-degree neighbors. These mechanisms remain hypotheses.

## Published comparison
The published ENZYMES result is **66.67% +/- 5.00%**, from Table 2, not a locally measured score. Quick mode uses a separate 5/3 nested protocol and a smaller grid, so it cannot demonstrate improvement over the paper. Even paper mode uses a new seed/projection generator and documented implementation differences.

## Runtime and memory
Feature time includes all eligible H/tau extraction computations. Kernel time includes squared distances and all gamma Gram matrices. Search time includes inner SVM fits and outer refits. Cache writes, plotting, dataset loading and report creation are excluded from the sum of compute times. Cold maps are recomputed on every run to keep comparisons interpretable; compressed maps and distance matrices are saved for inspection/reuse. Adaptive methods have three times the candidates in quick mode (81 versus 27) and therefore additional search cost. Timings are machine dependent. Memory is absolute process RSS sampled every 10ms, includes interpreter/data, and may miss short peaks; it is not isolated per-method allocation. Methods execute sequentially, so allocator retention may affect later RSS.

## Paired statistical analysis
Fold differences and descriptive 95% Student-t intervals, with exploratory paired t p-values, are saved in tables/paired_analysis.csv. Outer training sets overlap; fold differences are dependent, five folds give little power, and three comparisons plus hyperparameter search introduce multiplicity. These intervals and p-values do not establish statistical significance. Fold metrics, selected parameters, full validation grid and out-of-fold predictions are all saved.

## Figures
See figures/accuracy.png, macro_f1.png, computational_cost.png, temperature.png, wl_iterations.png, and confusion_matrices.png. All are 300dpi PNGs. captions.md explains error bars and the optimistic nature of best-grid inner-validation sensitivity curves. Confusion matrices use each graph's matching outer-test prediction.

## Limitations, conclusion and future work
This is a single dataset and seed. The baseline is independent and deviations are disclosed. A higher observed modified score is descriptive evidence only; equal or lower performance fails to support the accuracy hypothesis on this run. Extra tau flexibility incurs additional compute cost, so gains cannot be attributed to aggregation alone without more controlled studies. Repeat nested CV over prespecified seeds, test more datasets, verify exact Torch equivalence with shared projection arrays, and study a fixed-tau comparison before drawing stronger conclusions. Paper evaluation is optional and only its saved output, if present, establishes completion.

## Reproducibility and references
Run commands in README.md. Configuration, archive checksum, splits and dependency versions are recorded. Sources: paper above; https://gitlab.com/drti/swwl ; https://chrsmrrs.github.io/datasets/docs/datasets/ .
