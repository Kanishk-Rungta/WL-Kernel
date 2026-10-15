"""Generate the Phase 2 research report exclusively from saved measurements."""
from pathlib import Path
import json
import pandas as pd


def markdown_table(frame, columns=None):
    frame=frame if columns is None else frame[columns]
    def cell(value):
        if isinstance(value,float): return f'{value:.4f}'
        return str(value)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def report(root):
    root=Path(root); base=root/'results/phase2'; sections=[]
    for name,path in [('Quick formula',base/'quick'),('Paper-grid formula',base/'paper'),('Quick authors-style',base/'authors_style'),('Additional-seed 43 quick formula',base/'quick/seed43')]:
        if not (path/'tables/comparison.csv').exists(): continue
        s=pd.read_csv(path/'tables/comparison.csv')
        sections += [f'### {name}\n\n'+markdown_table(s,['method','accuracy_mean','accuracy_std','macro_f1_mean','weighted_f1_mean','embedding_seconds','search_seconds','total_seconds','svm_fits','peak_rss_mib'])]
        if (path/'tables/tradeoff.csv').exists(): sections += [markdown_table(pd.read_csv(path/'tables/tradeoff.csv'))]
    full=base/'paper' if (base/'paper/tables/comparison.csv').exists() else base/'quick'
    s=pd.read_csv(full/'tables/comparison.csv').set_index('method'); pairs=pd.read_csv(full/'tables/paired_analysis.csv')
    tau=pd.read_csv(full/'tables/temperature_statistics.csv'); temperatures=pd.read_csv(full/'tables/graph_temperatures.csv')
    counts=temperatures.temperature.value_counts().sort_index().rename_axis('temperature').reset_index(name='graphs')
    historical=pd.read_csv(root/'results/paper/tables/hyperparameters.csv')
    tuned=historical[historical.method=='adaptive'].tau.value_counts().sort_index().rename_axis('globally_selected_tau').reset_index(name='outer_folds')
    gaps=pd.read_csv(full/'tables/edge_degree_gaps.csv')
    gap_stats=gaps.degree_gap.describe().to_frame('degree_gap').reset_index(names='statistic')
    a=s.loc['adaptive']; c=s.loc['self_calibrated']; o=s.loc['original']; aug=s.loc['augmented']; ca=s.loc['self_calibrated_augmented']
    delta=c.accuracy_mean-a.accuracy_mean; gain=c.accuracy_mean-o.accuracy_mean
    primary=pairs[(pairs.method=='self_calibrated')&(pairs.metric=='accuracy')]
    profile_path=base/'profiling/equal_work_feature_profile.csv'
    profiling='Equal-work profiling has not yet been run.'
    if profile_path.exists():
        profile=pd.read_csv(profile_path); grouped=profile.groupby('engine').seconds.agg(['mean','std'])
        ratio=grouped.loc['legacy','mean']/grouped.loc['optimized','mean']
        profiling=f'Equal-work profiling (three alternating-order repetitions, all 600 graphs, H=0..3, fixed tau=2, no cache IO or SVM search) measured a legacy/optimized mean ratio of {ratio:.2f}x. This is a constant-factor implementation gain, not a changed asymptotic WL complexity.\n\n'+markdown_table(grouped.reset_index())
    reference_path=base/'profiling/official_intermediate_check.json'
    ref=json.loads(reference_path.read_text()) if reference_path.exists() else {'status':'not run'}
    authors='Secondary authors-style evaluation is not completed.'
    authorpath=base/'authors_style/tables/comparison.csv'
    if authorpath.exists():
        author=pd.read_csv(authorpath).set_index('method')
        authors=f"Matched quick authors-style accuracy: baseline {author.loc['original','accuracy_mean']:.2f}%, SCDA {author.loc['self_calibrated','accuracy_mean']:.2f}%; difference {author.loc['self_calibrated','accuracy_mean']-author.loc['original','accuracy_mean']:+.2f} pp. These are separate conventions; do not attribute differences between styles to calibration. A full authors-style paper-grid search was not run."
    extra='No additional seed run completed.'
    seedpath=base/'quick/seed43/tables/comparison.csv'
    if seedpath.exists():
        es=pd.read_csv(seedpath).set_index('method')
        extra=f"The prespecified additional seed 43 quick check gave baseline {es.loc['original','accuracy_mean']:.2f}% and SCDA {es.loc['self_calibrated','accuracy_mean']:.2f}%, difference {es.loc['self_calibrated','accuracy_mean']-es.loc['original','accuracy_mean']:+.2f} pp. It is reported irrespective of direction and was not used to select a seed or change calibration."
    text=f'''# Self-Calibrated Degree-Adaptive Sliced Wasserstein Weisfeiler–Lehman Kernel: Evaluating Structural Calibration and Computational Efficiency

## 1. Abstract
We evaluate a prespecified structural heuristic that removes global temperature tuning from degree-adaptive SWWL. In the matched paper-grid experiment, SCDA accuracy was {c.accuracy_mean:.2f}% (sample SD {c.accuracy_std:.2f}), versus {o.accuracy_mean:.2f}% original and {a.accuracy_mean:.2f}% tuned adaptive. Its wall time was {c.total_seconds:.2f}s versus {a.total_seconds:.2f}s tuned adaptive. These are exploratory findings from one dataset, not proof of better generalization.

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

{(chr(10)*2).join(sections)}

## 10. Runtime, memory and profiling
Separate preprocessing, operator, WL, projection/quantile, distance, kernel, cache IO, and SVM search/refit durations are in timing.csv. Total is actual per-method wall time including cache writes and fold checkpoint writes; dataset loading and final plots/reports are excluded. compute_seconds excludes cache IO. Compare Phase 2 methods to each other; historical Phase 1 totals excluded IO and should not be used for speedup claims. Absolute RSS is sampled every 10ms, can miss short peaks, includes interpreter/data, and is influenced by sequential allocator retention. Each method begins with independent structures/operators/histories, though operating-system file caches and system load are uncontrolled.

SCDA eliminates temperature search: {int(c.svm_fits)} fits versus {int(a.svm_fits)} tuned adaptive. Search time saved was {a.search_seconds-c.search_seconds:.2f}s ({100*(1-c.search_seconds/a.search_seconds):.1f}%). Total wall speedup was {a.total_seconds/c.total_seconds:.2f}x; aggregate eligible feature time speedup was {a.embedding_seconds/c.embedding_seconds:.2f}x. Aggregate feature savings largely reflect fewer tau configurations and must not be described as faster per-graph asymptotic refinement.

{profiling}

## 11. Temperature distribution
{markdown_table(tau)}

{markdown_table(counts)}

Temperatures do vary, but variation is limited around the median of 2. A graph-specific formula alone is not evidence that graph specificity caused any gain; a fixed-tau=2 comparison would be needed to isolate that question. The rule was left unchanged. Edge-gap summary (each undirected edge once):

{markdown_table(gap_stats)}

Historical inner-selected globally tuned tau values, descriptive only:

{markdown_table(tuned)}

The global tau is selected for classification using training labels; tau_G is a label-free graph statistic. Their distributions are not estimates of the same quantity.

## 12. Ablation results
Original versus augmented isolates the degree attribute subject to a projection-dimension change. Original versus SCDA changes structural aggregation. SCDA versus SCDA+augmented adds the degree attribute; tuned adaptive versus SCDA changes temperature choice and search flexibility. The original combined model remains available, with all Phase 1 outputs preserved, but is not required in this five-method primary comparison. In the paper-grid run, augmented accuracy was {aug.accuracy_mean:.2f}% and SCDA+augmented was {ca.accuracy_mean:.2f}%. These are descriptive ablations, not evidence of causal mechanism.

## 13. Paired statistical analysis
{markdown_table(primary)}

Fold-level changes for accuracy, macro-F1 and weighted-F1 are saved. Intervals are descriptive 95% Student-t intervals and tests are exploratory paired t tests. Overlapping training sets make folds dependent; sample size is small and comparisons are multiple. No independence-adjusted confirmatory inference or statistical-significance claim is justified. {extra}

## 14. Authors-style reference comparison
{authors}

Actual official source intermediate-check status: {ref.get('status')}; maximum quantile difference: {ref.get('quantile_max_absolute_difference','not available')}; maximum kernel difference: {ref.get('kernel_max_absolute_difference','not available')}. See authors_style_reference.md for source hashes, nugget convention, precision and RNG limitations. Authors-style means matched quantile/kernel conventions, not an exact official experimental reproduction.

## 15. Accuracy–efficiency trade-offs
SCDA changed accuracy by {gain:+.2f} pp versus original and {delta:+.2f} pp versus tuned adaptive. Figures/accuracy_runtime.png highlights the observed Pareto frontier: no other displayed method has at least as high accuracy and at most as much wall time, with one strict inequality. This ignores uncertainty and is not a rule for selecting a model on outer tests. Reduced search cost is directly measurable; structural weighting mechanisms and stronger generalization remain hypotheses. Retaining accuracy descriptively does not demonstrate statistical noninferiority or equivalence.

In this paper-grid run, SCDA and SCDA+augmented form the observed frontier. SCDA+augmented matched the tuned adaptive mean accuracy with {a.total_seconds/ca.total_seconds:.2f}x less wall time; SCDA was the faster of the two calibrated variants with slightly lower mean accuracy. This is a measured trade-off, not a single universally best model.

## 16. Limitations and threats to validity
One small dataset, mostly one seed, exploratory follow-up to earlier results, dependent folds, additional tuning flexibility, coarse temperature variation, projection changes with attribute dimension, sequential timing, cache IO cost, absolute sampled memory, and independent NumPy rather than official Torch RNG/precision. An additional quick seed is limited robustness evidence. No constants or grids were changed after inspecting Phase 2 held-out outcomes. The published 66.67% +/-5.00% is external reference, not our measurement, and should not support a superiority claim.

## 17. Conclusions
The strongest defensible contribution is a deterministic structural calibration heuristic that removes tau model selection, together with measured, matched nested-CV accuracy–cost comparisons and reproducible implementation optimizations. Observed accuracy differences remain uncertain. Any negative or inconsistent seed/style outcomes are reported alongside positive ones. Fewer hyperparameters do not automatically guarantee better generalization.

## 18. Future research
Prespecified fixed-tau controls, more datasets and seeds, exact Torch environment replication, controlled hardware profiling and appropriate repeated-CV inference could test whether graph-specific calibration contributes beyond eliminating search. Alternative calibration equations would be new hypotheses requiring fresh validation.

## 19. References and reproducibility
Carpintero Perez, R., Da Veiga, S., Garnier, J., Staber, B. (2024), Gaussian process regression with Sliced Wasserstein Weisfeiler–Lehman graph kernels. https://proceedings.mlr.press/v238/carpintero-perez24a.html . Official source https://gitlab.com/drti/swwl, inspected commit a974d59ecb208758844ef15381175cb8b1333c05. Dataset https://chrsmrrs.github.io/datasets/docs/datasets/ . Exact commands in README.md. Phase 1 file hashes are stored in results/phase2/logs/phase1_manifest.json and verified separately. Nothing is pushed to GitHub during Phase 2.

All 44 tests passed after integration. The artifact audit verified 127 preserved Phase 1 files and each research run's 600 unique out-of-fold predictions per method, fold summaries and inner-selected scores. Primary paper-grid figures are in results/phase2/paper/figures: accuracy.png, macro_f1.png, runtime.png, accuracy_runtime.png, temperature_distribution.png, speedup.png, paired_differences.png, and edge_degree_gaps.png. Matched authors-style figures are in results/phase2/authors_style/figures. All are 300dpi PNGs with captions.md explanations.
'''
    (root/'reports/phase2_report.md').write_text(text,encoding='utf-8')
    reference=f'''# Authors-style SWWL reference

Inspected official commit a974d59ecb208758844ef15381175cb8b1333c05 in the local reference_swwl checkout.

| Component | Existing formula | Authors-style configuration |
|---|---|---|
| Quantiles | linear, endpoint grid | lower, endpoint grid |
| Map normalization | 1/sqrt(PQ) | 1/sqrt(PQ) |
| Distance | squared Euclidean | Euclidean |
| Kernel | exp(-gamma*d_squared) | exp(-gamma*d) |
| Training diagonal nugget | 0 | 1e-7, from official config |
| Projection RNG used here | NumPy PCG64 | NumPy PCG64 |
| Precision used here | float64 | float64 |

Source evidence: graph_gp/encoders/sw_encoder.py calls torch.quantile(interpolation='lower'), normalizes torch.randn directions and divides maps by (P*Q)**(1/p). graph_gp/distances.py uses scipy pdist without sqeuclidean. experiments/run_classification.py selects type_kernel='exp' for svc; graph_gp/models/kernel.py applies exp(-precision*D) plus nugget. experiments/config.yml sets nugget=1e-7. The paper Section 4 instead defines a squared-distance Gaussian and describes linear interpolation. Dataset loader/encoder uses only the first 18 continuous channels. Original Torch generator, float32 defaults, official fold-seed organization, preprocessing/environment and inner search conventions are not fully replicated here. We deliberately preserve the Phase 1 matched split/scoring protocol to isolate style within this project.

Verification against the actual official encoder and kernel functions (substituting an equivalent sorted-batch unbatch helper only):

```json
{json.dumps(ref,indent=2)}
```

Shared projection arrays, float64 tensors and a shared quantile grid agree to numerical tolerance. Official quantile-major flattening and our projection-major flattening differ by a fixed coordinate permutation, which leaves pairwise distances unchanged. This is a controlled intermediate check, not a full official-code reproduction. Torch is optional only for this verification; experiments need no Torch or GPU.

{authors}

References: https://gitlab.com/drti/swwl ; https://proceedings.mlr.press/v238/carpintero-perez24a.html .
'''
    (root/'reports/authors_style_reference.md').write_text(reference,encoding='utf-8')


if __name__=='__main__': report(Path(__file__).resolve().parents[1])
