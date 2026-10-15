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
This run is **paper**. Full configuration and software versions are saved in logs/config.json. Identical stratified outer and inner splits are saved in logs/splits.json. Hyperparameters are selected solely by mean inner accuracy, with first-candidate deterministic ties. Outer tests are evaluated once for the selected configuration. The projection seed is fixed and directions are identical whenever dimensions match. Label-independent graph-local maps can safely be computed before splitting. No parameter was adjusted after examining outer outcomes.

Quick: 5 outer folds, 3 inner folds; H=[1,2,3], gamma=[0.001,0.01,0.1], C=[0.1,1,10], adaptive tau=[0.5,2,5], P=Q=20. Paper mode: 10/5 folds with H=[0,1,2,3], gamma=[0.0001,0.001,0.01,0.1], C=10**[-3,...,3], tau=[0.5,1,2,5]. Paper-compatible describes fold counts and grids, not identical random splits or official-code reproduction.

## Baseline, modified performance, and ablations
The following CSV table gives outer-fold means and sample standard deviations (accuracy in percent; F1 on 0-1 scale), absolute and relative changes from baseline, timings, and sampled process memory:

```csv
method,accuracy_mean,accuracy_std,macro_f1_mean,macro_f1_std,weighted_f1_mean,weighted_f1_std,embedding_seconds,kernel_seconds,search_seconds,total_seconds,peak_rss_mib,svm_fits,candidates_per_fold,accuracy_difference_pp,relative_improvement_percent
original,64.66666666666666,4.067334492827361,0.6467538242806441,0.0373039796050701,0.6467538242806441,0.0373039796050701,1.7401946999889333,0.2553634000068996,48.57738509999763,50.57294319999346,230.85546875,5610,112,0.0,0.0
adaptive,66.33333333333333,5.258737584977435,0.6610740455488507,0.0507677587613394,0.6610740455488507,0.0507677587613394,11.564272400035406,1.191247700000531,201.36570910000592,214.12122920004185,367.29296875,22410,448,1.6666666666666714,2.5773195876288737
augmented,66.0,5.9421075364981935,0.6607308060639598,0.0582331452843044,0.6607308060639598,0.0582331452843044,1.6176679999771295,0.2323463000066112,46.98363259999314,48.83364689997688,236.75,5610,112,1.3333333333333428,2.061855670103108
combined,66.16666666666666,5.668300418082089,0.6597840574657441,0.0530082065028181,0.6597840574657441,0.0530082065028181,11.618424399959622,1.252537599983043,205.21556549999517,218.0865274999378,369.41015625,22410,448,1.5,2.31958762886598

```

Measured baseline accuracy is 64.67% +/- 4.07% (sample SD), macro-F1 0.6468. Its numerical difference from the published mean is -2.00 percentage points, without protocol equivalence. adaptive: 66.33% +/- 5.26%, macro-F1 0.6611, accuracy delta +1.67 pp. augmented: 66.00% +/- 5.94%, macro-F1 0.6607, accuracy delta +1.33 pp. combined: 66.17% +/- 5.67%, macro-F1 0.6598, accuracy delta +1.50 pp. adaptive descriptive paired 95% interval: [-1.09, 4.42] pp. augmented descriptive paired 95% interval: [-1.98, 4.65] pp. combined descriptive paired 95% interval: [-1.80, 4.80] pp. The observed changes do not establish a reliable improvement: the descriptive accuracy intervals overlap zero. This run gives no adequate evidence of statistical significance.

The method with highest observed mean accuracy is **adaptive** (66.33%). This ranking describes these folds and does not select a deployable model using test performance. A/B isolate weighting; A/C examine degree augmentation; D combines both. Different attribute dimensions change the random projection geometry, which is a limitation of the augmentation comparison. A possible benefit of weighting is preservation of similar-degree neighborhoods; a possible harm is suppressing useful dissimilar-degree neighbors. These mechanisms remain hypotheses.

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
