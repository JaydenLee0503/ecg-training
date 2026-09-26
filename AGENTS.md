# Project instructions for coding assistants

Read [architects/SESSION_HANDOFF.md](architects/SESSION_HANDOFF.md) at the start of a new session. It records the current research state, verified results, restart checks, and the next proposed experiment. Follow links to the detailed reports before making scientific claims.

## Working rules

- ACS / OMI work belongs in `experiments/acs_omi_vmd_wst_vqc/`. Read its `SESSION_HANDOFF.md` and keep its code, tests, protocols, reports, log, data and results there. Original ECGData experiments retain their current layout; shared numerical methods remain in `ecgvmd/`.
- The project compares VMD + VQC with standard WST + VQC for ECG classification. The user wants every experiment documented, including negative results, exclusions, splits, seeds, settings, runtimes, predictions, and uncertainty.
- Check saved completion markers and artifacts before starting expensive computation. A new chat or PC restart is not a reason to retrain a completed experiment.
- Keep existing results and frozen protocols intact. Use a separate output directory and a new documented protocol for a changed experiment.
- Split by patient. Fit feature selection, scaling, and model tuning within the training/validation partitions. Keep final test patients out of model selection.
- Use the corrected patient-grouped reports as the reference. Older row-grouped tables in README.md and QUANTUM_STAGE.md are historical and must not be presented as patient-independent results.
- Report all initialization seeds and the relevant classical controls. Do not select a favorable seed or imply that an unverified paper's accuracy is directly comparable.
- Numerical checks do not establish that software has no bugs. Current results do not establish quantum advantage or superiority of VMD over WST.
- The handoff is a dated record, not a new approval requirement. Follow the user's current instructions and the actual environment permissions; do not invent additional confirmation steps.
- After substantial work, update SESSION_HANDOFF.md and the relevant experiment report/log with what actually finished, what remains pending, and any failures.

## Environment

The existing Python environment is at `/home/jaydenlee/venvs/test-ecg-training/bin/python`. After a restart, `python` may not be on PATH. Check the interpreter before installing anything. `python3` is sufficient for the standard-library artifact checksum check in the handoff.

Large generated files in `results/` and `features/`, including experiment-local directories, are gitignored. Raw ACS archives in `experiments/acs_omi_vmd_wst_vqc/data/` are also gitignored. Documentation in Git does not mean a fresh clone contains the saved models, data or feature archives.
