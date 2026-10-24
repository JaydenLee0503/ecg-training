# ECG-06 — LFCC/Swin epoch selection and dropout

Status: **complete**.

Completed 90 inner fits and 30 outer trials, with all three seeds. The primary selected procedure and the fixed dropout arm are reported separately. This is not a larger-model or raw-waveform experiment.

## Files

[Open the clickable file index](FILES.html), or use the canonical links below.

| Resource | Original location |
|---|---|
| `report` | [attention method/reports/ecgdata_lfcc_swin_regularization_v2](../../../attention%20method/reports/ecgdata_lfcc_swin_regularization_v2) |
| `artifacts` | [attention method/results/ecgdata_lfcc_swin_regularization_v2](../../../attention%20method/results/ecgdata_lfcc_swin_regularization_v2) |
| `protocol.json` | [attention method/ecgdata_regularization_protocol_v2.json](../../../attention%20method/ecgdata_regularization_protocol_v2.json) |
| `protocol.md` | [attention method/ecgdata_regularization_protocol_v2.md](../../../attention%20method/ecgdata_regularization_protocol_v2.md) |
| `code/run_ecgdata_regularization.py` | [attention method/scripts/run_ecgdata_regularization.py](../../../attention%20method/scripts/run_ecgdata_regularization.py) |
| `code/audit_ecgdata_regularization.py` | [attention method/scripts/audit_ecgdata_regularization.py](../../../attention%20method/scripts/audit_ecgdata_regularization.py) |
| `code/ecgdata_regularization.py` | [attention method/attention_ecg/ecgdata_regularization.py](../../../attention%20method/attention_ecg/ecgdata_regularization.py) |
| `code/regularized_swin.py` | [attention method/attention_ecg/regularized_swin.py](../../../attention%20method/attention_ecg/regularized_swin.py) |

Keep commands relative to the repository root as documented in the original report. Scripts, frozen protocols, checkpoints and data have not moved. Shared code is intentionally linked rather than copied.

Large generated artifacts are gitignored and available only when the original local data are present. A fresh clone may therefore have missing artifact targets. The indexes use ordinary relative links and do not require symbolic-link support.

[All studies](../../README.md).
