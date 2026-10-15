# Execution status

Completed locally: 18 pytest checks, smoke for all four methods, full ENZYMES quick (5x3 nested CV) and full paper-grid (10x5 nested CV), seed 42, all four methods. Both protocols have six real-data figures, configurations, splits, predictions, tables, paired analyses and reports. Artifact audit verified exactly 600 unique OOF predictions per method and agreement between folds, means and inner-selected scores.

Quick: baseline 65.50%, adaptive 66.17%, augmented 66.33%, combined 66.50%.
Paper-grid: baseline 64.67%, adaptive 66.33%, augmented 66.00%, combined 66.17%.

Observed changes are small and all descriptive paired accuracy intervals span zero. No reliable statistical improvement is established. Single seed/dataset; independent paper-formula reproduction, not exact authors-code execution. See reference_review.md. No requested experiment mode remains unexecuted. Repeated-seed/multi-dataset validation remains future work.
