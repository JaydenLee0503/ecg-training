# GPU extraction stopped at the VMD convergence safeguard — 2026-09-25 EDT

The full run stopped at **2026-09-26 01:29:28 UTC (21:29 EDT, September 25)**.
The process exited with code 1. Status is `failed`, not running or complete.
It saved **1,795 / 17,905 ECGs (10.03%)**; **16,110 remain pending**.
No current completion-time estimate applies while extraction is stopped.

The only final capped lead in the active 16-record batch was **01985 / V5**.
This record belongs to the fit partition. It exhausted all declared restart
attempts: 2,000, 4,000, 8,000, 16,000 and 32,000 iterations, at unchanged tolerance
1e-7. The other 191 leads in that batch have final uncapped attempts. The batch
failed before descriptor/WST outputs were yielded, so none of those 16 records
has a completed feature checkpoint. Their attempt logs and failure traceback
are retained; no record was silently excluded and no capped result was accepted.

## Integrity verification

All 1,795 completed checkpoints were reread and passed identity, feature-file
hash, feature count/precision, finite-value and saved attempt-log hash checks.
They comprise 115 imported CPU records and 1,680 GPU VMD / CPU WST records,
including the 16-record recovery batch. The frozen code/protocol/environment
and parent/source/split manifests also passed the normal read checks. See the
[stop verification JSON](gpu_extraction_stop_2026-09-25.json).

## CPU reference diagnostic

A separate diagnostic was declared before computation in
`results/vmd_cap_01985_cpu_diagnostic_v1/protocol.json`. It reloaded the verified
source record and ran the unchanged CPU `vmd_batch` solver on lead V5 alone,
from the same initialization, at the same final 32,000-iteration cap. Settings:
K=8, alpha=2000, tau=0, dc=true, init=1, tol=1e-7, fs=500 Hz, float64 input.
The probe did not change the production protocol or resume extraction.

The CPU attempt also returned **32,000 iterations, capped=true**, in **16.466
seconds**. Modes and centre frequencies were finite. This reproduces the
iteration-cap symptom on CPU; switching back to CPU alone does not address it.
It does not prove all GPU arithmetic correct or determine whether a higher cap
will converge. See the [diagnostic JSON](vmd_cap_01985_cpu_diagnostic_2026-09-25.json)
for settings, identities, environment, source/code hashes and timing. The local
result directory retains the probe source snapshot and CPU mode/frequency arrays.

## Next step

Investigate convergence on this single fit lead in a separate diagnostic, for
example a bounded higher-cap check with the tolerance and all other scientific
settings unchanged. Do not repeatedly restart the full frozen run or silently
change its cap/tolerance, exclude this record, or merge it with a different
protocol. If a retry-policy amendment is justified, declare a new protocol and
output manifest and explicitly reuse the 1,795 verified checkpoints. The saved
features do not need to be thrown away or recomputed simply because this run
stopped. Full extraction, bundling and all ACS classifier training remain pending.
