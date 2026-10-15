# Reference review

Inspected official GitLab main commit a974d59ecb208758844ef15381175cb8b1333c05 (cloned alongside project).

- graph_gp/encoders/wl_encoder.py: WLEncoder delegates to PyG WLConvContinuous and concatenates original plus all iterations.
- graph_gp/encoders/embedding_encoder.py: SWWL_Encoder slices original continuous attributes and projects the entire concatenated node vector.
- graph_gp/encoders/sw_encoder.py: normalized Gaussian random projections, evenly spaced endpoint quantiles, lower interpolation, (P*Q)^(-1/2) scaling.
- graph_gp/distances.py: distance_matrices_swwl returns ordinary Euclidean map distances.
- graph_gp/models/kernel.py: custom_kernel_exp uses exp(-precision*D); custom_kernel_rbf squares D. Classification script chooses exp for svc.
- experiments/datasets_classification.py: ENZYMES dimensionality fixed at 18, with categorical appended loader channels excluded by encoder slicing.
- experiments/run_classification.py and scripts/script_classification.sh: H=0..3, P=Q=20, nested tuning of H/gamma/C, shared outer splits, seeds 0..9.
- Paper Sections 4/5: squared-distance Gaussian, linear quantile interpolation, original features only, 10 outer/5 inner stratified CV, gamma=10^-4..10^-1, C=10^-3..10^3. Table 2 published ENZYMES accuracy 66.67 +/- 5.00.

Decision: implement requested paper-formula baseline independently in NumPy/SciPy, explicitly disclose code discrepancies. Do not call it a bitwise reproduction of the authors' implementation. No source code copied. Source links are in README/report.
