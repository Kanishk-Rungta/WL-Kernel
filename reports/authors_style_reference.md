# Authors-style SWWL reference

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
{
  "status": "passed",
  "torch_version": "2.5.1+cpu",
  "quantile_max_absolute_difference": 2.220446049250313e-16,
  "kernel_max_absolute_difference": 0.0,
  "encoder_sha256": "893a61f7cd932cf167350680a87f77664fff4e6b6dde45a2d0253424464e5d97",
  "kernel_source_sha256": "79d837d21953c3a1f06eed24bd13ee8d646cf962ac4a404e9cbda50e38c73f6b",
  "scope": "Actual official SWEncoder and custom_kernel_exp with shared directions, float64 and shared quantile grid; equivalent unbatch helper. Does not validate Torch RNG equivalence, original precision or full official training environment."
}
```

Shared projection arrays, float64 tensors and a shared quantile grid agree to numerical tolerance. Official quantile-major flattening and our projection-major flattening differ by a fixed coordinate permutation, which leaves pairwise distances unchanged. This is a controlled intermediate check, not a full official-code reproduction. Torch is optional only for this verification; experiments need no Torch or GPU.

Matched quick authors-style accuracy: baseline 66.67%, SCDA 67.00%; difference +0.33 pp. These are separate conventions; do not attribute differences between styles to calibration. A full authors-style paper-grid search was not run.

References: https://gitlab.com/drti/swwl ; https://proceedings.mlr.press/v238/carpintero-perez24a.html .
