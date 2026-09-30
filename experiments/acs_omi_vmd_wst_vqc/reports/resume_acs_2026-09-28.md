# Resume ACS extraction after sleep or restart

## Latest pause — September 29 EDT

The user requested a pause for sleep. Extraction finished its batch and exited
successfully at **2026-09-29T04:41:06 UTC**, leaving **12,916/17,905 saved ECGs**
and **4,989 pending**. Its status is `interrupted` with `stop_signal: 15` and no
active records. The downstream workflow also exited. Its `stopped` / `failed`
marker and `Extraction stopped: interrupted` traceback are the expected response
to this pause. No classifier has been fitted and official test data are reserved.

The final full integrity check passed at **04:57:05 UTC**: all **12,916 saved
checkpoints verified**, and the reused parent feature files remain unchanged.
All work for this pause is finished; the verifier has also exited.
See the [saved pause record](pause_acs_2026-09-29.json).

Tomorrow, check the interpreter, temporary GPU packages and live process/status
state, then resume the **full extraction command below**. It checks and reuses
the saved checkpoints. After its status changes to `running`, restart the same
`finish_development256k` workflow command below. Do not start the watcher while
the old extraction status still reads `interrupted`; it would stop immediately.

The previous full session processed 9,200 new ECGs at about 2,662 ECGs/hour.
That suggests about 1 hour 52 minutes of extraction left at the old rate, plus
startup checks and subsequent verification/model work. Do not repeat the completed
one-batch recovery test or prepare a new run solely because the PC/chat restarted.

## Saved state

- New active run: `results/omi_v1_gpu_retry256k/`.
- Manifest SHA-256: `fc080f2481d1cb1c671f99f78876391d1d1af5cba7bd74a3c2040182cb224474`.
- `preparation.json`: complete for 17,905 eligible development ECGs.
- The controlled one-batch recovery finished at 2026-09-29T00:50:12 UTC with
  **3,716 records** saved. Verification passed at 00:59:27 UTC: **14,189 pending**;
  all 3,700 reused parent feature files are unchanged.
- Latest `verification.json`: **12,916 verified / 4,989 pending**, passed at
  2026-09-29T04:57:05 UTC after the requested pause.
- Full continuation ran on September 28–29 and is now paused at 12,916 saved
  records. The 3,716 count above records the earlier completed recovery check.
  Read `extraction_status.json` for the authoritative current state and count.
- The preceding 128k run and its 3,700 completed records remain preserved.

The new retry limit is 256,000. It changes no other signal-processing setting.
Record 04124/V5 converged at iteration 128,234 on the unchanged CPU and GPU solvers;
their maximum mode difference was 5.7421e-13. Full details and the trace are in
`vmd_04124_diagnostic_2026-09-27.md`.

## Start safely

From the repository root, first inspect the current status and confirm no extraction
process is active. The data, GPU packages and saved run must be available. Use the
existing interpreter and keep the computer awake during extraction.

```bash
python3 -B - <<'PY'
from pathlib import Path
import json
p = Path("experiments/acs_omi_vmd_wst_vqc/results/omi_v1_gpu_retry256k/extraction_status.json")
print(json.loads(p.read_text()) if p.exists() else "Prepared and verified; extraction has not started")
PY
```

The one-batch recovery and full checkpoint verification below already passed.
These are historical commands; do not repeat the one-batch test just because a
new chat starts:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m experiments.acs_omi_vmd_wst_vqc.scripts.extract_gpu_retry256k extract --max-batches 1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m experiments.acs_omi_vmd_wst_vqc.scripts.extract_gpu_retry256k verify
```

After confirming that no extraction process is active, an operationally
interrupted run can continue with the same manifest and no batch limit:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m experiments.acs_omi_vmd_wst_vqc.scripts.extract_gpu_retry256k extract
```

This resumes verified checkpoints and stops on another convergence failure. Do
not start a competing writer. Do not change tolerance, exclude a capped record or
reuse the old run directory for a changed protocol. The already authorized
downstream workflow can wait while extraction is running, or start after it
completes. Check `results/omi_development_workflow_v2/status.json` and the live
processes first to avoid launching a duplicate:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jaydenlee/venvs/test-ecg-training/bin/python -B -m experiments.acs_omi_vmd_wst_vqc.scripts.finish_development256k
```

It verifies and bundles all features before model fitting. It trains the fixed
comparison and reports patient-cluster uncertainty only after successful extraction.
Official test records remain reserved.

## Temporary GPU dependencies

The original base Python environment was not changed. The pinned CuPy 14.2.0,
cuda-pathfinder 1.8.2 and NumPy 2.5.2 target install is at
`/tmp/acs-gpu-deps-v1`; the extraction module adds this path when it starts. If a
full PC reboot removed `/tmp`, restore those exact versions to that target using
the same Python environment before starting GPU work.
