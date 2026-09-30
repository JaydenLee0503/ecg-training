# ACS bounded VMD retry amendment — 2026-09-27

The original `omi_v1_gpu` run remains frozen at its numerical stop. This separate
run uses `acs_omi_protocol_v1_retry128k.json` and
`gpu_extraction_retry128k_v1.json`, under `results/omi_v1_gpu_retry128k/`.

## Declared change and evidence

- Append 64,000 and 128,000 iteration retries to the original limits. Each retry
  starts from the same initialization. Tolerance remains 1e-7; all other scientific
  inputs, algorithms, features, split and classifier settings remain unchanged.
- The failed fit ECG 01985/V5 converged at iteration 67,827. The saved CPU and GPU
  diagnostic features, physical inputs and convergence metadata agreed exactly.
- Restore the original temporary CuPy 14.2.0, cuda-pathfinder 1.8.2 and NumPy 2.5.2
  dependencies after restart. The base Python environment is unchanged.
- Revalidate all 16 original fit-pilot ECGs plus 01985 on the current GPU. The
  17-record check completed in 127.658 seconds; all comparison checks passed,
  and WST was checked against Kymatio for all 17 ECGs. CUDA reports the same
  hardware/runtime/driver fingerprint as the original run.
- Copy the 1,795 verified original checkpoints plus the completed diagnostic ECG
  without recomputing their features. Preserve source feature bytes and attempt
  logs; record parent identities and hashes in the new manifest.

## Execution and failure handling

Use the existing 16-ECG GPU batches, FP64 solver and CPU WST/descriptors. Check the
new manifest, device validation, imports and all completed records before resume.
Run one batch and verify clean interruption/recovery before full continuation.
The final 128,000 limit remains bounded: another nonconvergence stops extraction,
retains every attempt and does not silently exclude records or relax tolerance.

Do not process official-test ECGs. After all 17,905 development records complete,
verify both feature arms, finite values, convergence, names, identities and split
coverage using `scripts/bundle_gpu.py`. Only a complete bundle can enter training.

## Fixed downstream comparison

The user authorized continuing ACS extraction and the planned classifier comparison
on 2026-09-27. This supersedes historical extraction-only execution scope; the
scientific settings remain those declared in the original ACS protocol.

`scripts/train_models.py` will fit train-only mRMR/scaling, four classical controls
(two feature arms × logistic/KNN) and all six VQCs (two arms × seeds 0/1/2).
`training.py` adds atomic epoch recovery to a local copy of the frozen VQC fit
loop. A synthetic parity test verifies identical probabilities and losses against
the original implementation after interruption and resume. The validation labels
do not enter training, epoch selection or threshold selection.

`scripts/report_models.py` will retain all per-seed probabilities, fixed-threshold
metrics, constant-prevalence/always-negative control results, and 2,000 paired
patient-cluster bootstrap draws with seed 20260925. VQC summaries average seed
metrics, not probabilities. Source snapshots, hashes, warnings, histories, optimizer
checkpoints and runtimes accompany the models. No final-test inference is authorized
by this development protocol.
