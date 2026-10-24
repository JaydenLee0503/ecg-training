# Historical ECGData records

These records preserve earlier work, including failures and engineering checks. Their classifier estimates use historical row groups and must not be presented as patient-independent validation. E3 is setup only; E18b belongs to E18.

## Numbered log entries

| ID | Experiment | Status |
|---|---|---|
| E1 | [Readiness check](e01_readiness_check/README.md) | historical |
| E2 | [Angle wrap-around diagnostic](e02_angle_wraparound/README.md) | historical |
| E3 | [Environment](e03_environment_setup/README.md) | setup only |
| E4 | [Kernel correctness and throughput](e04_kernel_correctness_throughput/README.md) | historical |
| E5 | [First stage-2 probe](e05_invalid_kernel_probe/README.md) | invalid |
| E6 | [Gram geometry sweep](e06_gram_geometry/README.md) | historical |
| E7 | [Bandwidth CV sweep](e07_bandwidth_cv/README.md) | historical |
| E8 | [Product-state factorisation](e08_product_state_factorization/README.md) | historical |
| E9 | [Deeper bandwidths + tuned rivals](e09_bandwidth_and_classical_rivals/README.md) | partially failed |
| E10 | [IQP geometry, tuned rivals, full-scale product kernel](e10_iqp_geometry_and_rivals/README.md) | historical |
| E11 | [Entangled IQP vs classical rivals](e11_entangled_iqp_comparison/README.md) | historical |
| E12 | [Noise floor](e12_seed_noise_floor/README.md) | historical |
| E13 | [VQC throughput](e13_vqc_throughput/README.md) | historical |
| E14 | [VQC first smoke test](e14_collapsed_vqc_smoke/README.md) | failed model collapse |
| E15 | [VQC learning-rate and step-budget sweep](e15_vqc_learning_rate_budget/README.md) | interrupted |
| E16 | [VQC training curve at full scale](e16_interrupted_vqc_training_curve/README.md) | interrupted |
| E17 | [The linear-head control](e17_linear_head_control/README.md) | historical |
| E18 | [Capacity probe (overfit 96 samples)](e18_vqc_capacity_probe/README.md) | historical |
| E19 | [The step budget, done properly](e19_vqc_full_training_curve/README.md) | historical |
| E20 | [Stage 4, the five-fold](e20_vqc_fivefold/README.md) | historical |
| E21 | [Bandwidth nested in-fold](e21_nested_kernel_bandwidth/README.md) | historical |
| E22 | [IQP nested at full scale](e22_nested_iqp_full_scale/README.md) | historical |
| E23 | [Register width: what widening actually costs](e23_qubit_width_cost/README.md) | historical |
| E24 | [Where the VQC's deficit actually comes from](e24_vqc_readout_probe/README.md) | historical |
| E25 | [The full metric panel, and a paired bootstrap](e25_full_metric_panel/README.md) | historical |

## Earlier unnumbered workflows and audit

These five groups organize related older runs; they are not a count of every exploratory fit.

| ID | Experiment | Status |
|---|---|---|
| H-01 | [Early classical feature, classifier and rhythm controls](h01_classical_feature_baselines/README.md) | historical |
| H-02 | [VMD alpha and convergence-budget sweep](h02_vmd_alpha_convergence/README.md) | historical |
| H-03 | [Historical WST/VMD comparison and temporal-pooling controls](h03_wst_vmd_row_grouped/README.md) | historical |
| H-04 | [Historical WST envelope-descriptor controls](h04_wst_descriptor_controls/README.md) | historical |
| H-05 | [Scattering numerical and KNN split audit](h05_scattering_and_split_audit/README.md) | audit complete |

[Shared early outputs](shared_outputs/README.md) retain exports without a verified one-to-one E-number association. [Complete catalog](../README.md).
