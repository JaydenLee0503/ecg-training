# ECG-09 representation comparison

Primary evaluation: identical 1,620 v1 windows; values mean across seeds 0/1/2 except deterministic logistic.

| Arm | Window accuracy | Window macro-F1 | Patient-vote accuracy | ARR / CHF / NSR recall |
|---|---:|---:|---:|---|
| lfcc_swin | 86.83% | 0.8390 | 95.83% | 90.94% / 76.11% / 84.81% |
| lfcc_logistic | 77.65% | 0.7462 | 90.00% | 78.23% / 72.00% / 80.83% |
| band_swin | 91.05% | 0.8846 | 98.33% | 95.35% / 79.44% / 89.26% |
| band_logistic | 86.67% | 0.8345 | 95.00% | 89.06% / 76.67% / 88.61% |
| raw_cnn | 63.97% | 0.5795 | 67.50% | 71.32% / 33.00% / 70.19% |

band_swin minus lfcc_swin: macro-F1 +0.0457, paired 95% interval [+0.0202, +0.0800].

raw_cnn minus lfcc_swin: macro-F1 -0.2595, paired 95% interval [-0.3273, -0.1933].

band_swin minus band_logistic: macro-F1 +0.0501, paired 95% interval [+0.0060, +0.0992].

band_logistic minus lfcc_logistic: macro-F1 +0.0884, paired 95% interval [+0.0478, +0.1347].

All-window results, all seeds, class recalls and uncertainty: [analysis.json](analysis.json).
35 new fits; summed fit runtime 6343.7 s; no data exclusions.
Pointwise bootstrap intervals condition on saved fits and do not include retraining uncertainty. Different representation input dimensions and raw CNN capacity are disclosed in PROTOCOL.md. ECGData contains only 80 patients and source/diagnosis confounding; no external or clinical validation.
