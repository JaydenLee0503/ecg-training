# ECG-07 all-window LFCC/Swin — results

Generated from saved out-of-fold predictions. Swin rows are means over seeds 0/1/2; logistic is deterministic. Intervals: 2,000 paired patient-bootstrap draws (seed 20260925), pointwise, conditional on saved predictions, without retraining or multiplicity adjustment.

## Matched evaluation on the 1,620 v1 windows (primary)

| Arm | Window accuracy | Window macro-F1 | Patient-vote accuracy | Patient-vote macro-F1 | Window recall ARR / CHF / NSR |
|---|---:|---:|---:|---:|---|
| all_swin | 86.83% | 0.8390 | 95.83% | 0.9425 | 90.94% / 76.11% / 84.81% |
| all_logistic | 77.65% | 0.7462 | 90.00% | 0.8753 | 78.23% / 72.00% / 80.83% |
| v1_swin | 83.35% | 0.8006 | 94.17% | 0.9269 | 88.82% / 72.00% / 78.24% |
| v1_logistic | 76.48% | 0.7342 | 87.50% | 0.8517 | 77.19% / 70.67% / 79.44% |

| Paired difference | Metric | Observed | 95% interval | Draws > 0 |
|---|---|---:|---:|---:|
| all_swin − v1_swin | window_macro_f1 | +0.0384 | [+0.0153, +0.0602] | 99.9% |
| all_swin − v1_swin | window_accuracy | +3.48 pp | [+1.58 pp, +5.38 pp] | 100.0% |
| all_swin − v1_swin | patient_vote_macro_f1 | +0.0155 | [-0.0272, +0.0559] | 78.0% |
| all_swin − v1_swin | patient_vote_accuracy | +1.67 pp | [-1.25 pp, +5.00 pp] | 83.7% |
| all_logistic − v1_logistic | window_macro_f1 | +0.0120 | [-0.0053, +0.0301] | 91.5% |
| all_logistic − v1_logistic | window_accuracy | +1.17 pp | [-0.31 pp, +2.87 pp] | 93.5% |
| all_logistic − v1_logistic | patient_vote_macro_f1 | +0.0236 | [-0.0076, +0.0711] | 85.8% |
| all_logistic − v1_logistic | patient_vote_accuracy | +2.50 pp | [+0.00 pp, +6.25 pp] | 87.1% |
| all_swin − all_logistic | window_macro_f1 | +0.0928 | [+0.0424, +0.1479] | 100.0% |
| all_swin − all_logistic | window_accuracy | +9.18 pp | [+4.69 pp, +13.70 pp] | 100.0% |
| all_swin − all_logistic | patient_vote_macro_f1 | +0.0672 | [+0.0005, +0.1451] | 97.5% |
| all_swin − all_logistic | patient_vote_accuracy | +5.83 pp | [+0.00 pp, +12.50 pp] | 97.5% |
| v1_swin − v1_logistic | window_macro_f1 | +0.0664 | [+0.0224, +0.1143] | 99.8% |
| v1_swin − v1_logistic | window_accuracy | +6.87 pp | [+2.75 pp, +11.27 pp] | 99.9% |
| v1_swin − v1_logistic | patient_vote_macro_f1 | +0.0752 | [+0.0055, +0.1589] | 98.7% |
| v1_swin − v1_logistic | patient_vote_accuracy | +6.67 pp | [+0.42 pp, +13.33 pp] | 98.0% |

Valid bootstrap draws: 2000/2000.

## All 21,222 held-out windows

| Arm | Window accuracy | Window macro-F1 | Patient-vote accuracy | Patient-vote macro-F1 | Window recall ARR / CHF / NSR |
|---|---:|---:|---:|---:|---|
| all_swin | 86.06% | 0.8290 | 95.42% | 0.9335 | 90.89% / 74.61% / 82.75% |
| all_logistic | 78.07% | 0.7505 | 88.75% | 0.8642 | 78.28% / 72.62% / 82.06% |

| Paired difference | Metric | Observed | 95% interval | Draws > 0 |
|---|---|---:|---:|---:|
| all_swin − all_logistic | window_macro_f1 | +0.0785 | [+0.0290, +0.1308] | 100.0% |
| all_swin − all_logistic | window_accuracy | +7.99 pp | [+3.71 pp, +12.40 pp] | 100.0% |
| all_swin − all_logistic | patient_vote_macro_f1 | +0.0693 | [+0.0037, +0.1417] | 98.1% |
| all_swin − all_logistic | patient_vote_accuracy | +6.67 pp | [+0.83 pp, +13.75 pp] | 98.3% |

Valid bootstrap draws: 2000/2000.

## Fits

20 fits; summed fit time 4591.2 s. Per-fit rows: [fits.csv](fits.csv). Per-seed metrics: [metrics.csv](metrics.csv). Full analysis: [analysis.json](analysis.json).
