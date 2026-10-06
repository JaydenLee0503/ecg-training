# Balanced ACS OMI training

This separate experiment was authorized on 2026-10-03. Its location follows the
user's explicit request. The original ACS and ECGData experiments remain intact.

**Complete and verified on 2026-10-04.** All 14 fits and the paired report are
saved. Read the [results summary](reports/v1/SUMMARY.md) and
[SESSION_HANDOFF.md](SESSION_HANDOFF.md). No retraining is needed.

## Plan

1. Reuse the completed ACS patient split, VMD/WST features and fitted mRMR/scalers.
2. Keep all 14,324 original fit ECGs. Append 12,490 sampled positive examples
   with replacement, giving 13,407 examples per class. Sampling seed: 20261003.
3. Train VMD/WST VQC with seeds 0/1/2 and matched logistic/KNN controls, removing
   class weights. Retain all other frozen classifier settings and 40 epochs.
4. Extract LFCC sequences using the existing ACS adapter: 37 frames × 12 leads
   × 20 coefficients. Fit normalization on unique original fit ECGs only.
5. Train the existing temporal Swin architecture from scratch with seeds 0/1/2,
   40 epochs and unweighted cross entropy. Add a temporal-mean/std LFCC logistic
   control. All new classifiers use the same oversampled fit index list.
6. Evaluate once at epoch 40 and threshold 0.5 on the original 3,581 validation
   ECGs. Report every seed, AP, ROC AUC, balanced accuracy, sensitivity,
   specificity, precision, F1, accuracy and confusion counts. Use 2,000 paired
   patient-bootstrap draws; official test ECGs remain reserved.

## Interpretation

This is an exploratory follow-up informed by the original validation results.
Repeated ECGs add no independent patients. Original VQC/logistic already used
balanced class weights. Oversampling changes minibatches and the optimizer budget:
40 VQC epochs now take 33,520 updates instead of 17,920. Keeping logistic C=1
also changes its effective regularization when rows are duplicated. The comparison
therefore does not isolate sampling at equal computation/regularization.

LFCC + Swin uses a different representation and model from VMD/WST + VQC. Its
comparison measures complete pipelines. The implementation is a one-dimensional
temporal adaptation of Swin, not an image model pretrained on another dataset.
Higher OMI sensitivity may accompany lower overall accuracy. Intervals condition
on saved predictions; they exclude training/sampling uncertainty. No external
validation, clinical validity or quantum advantage is established by this run.

## Run and resume

Use the existing Python interpreter. The runner uses the separate existing
PyTorch installation in `attention method/.venv/torch-cu128/` and the original
ACS dependency target `/tmp/acs-gpu-deps-v1/`. CUDA commands require GPU access.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jaydenlee/venvs/test-ecg-training/bin/python -B 'correct balance training set OMI/run.py' all
```

Run the same command after an interruption. Completed artifacts are checked and
reused; VQC and Swin resume saved epochs and LFCC extraction resumes saved batches.
Inspect `results/v1/status.json`, `results/v1/balanced.log`, `results/v1/run.log`
and live processes before launching another process. The runner captures its
stdout/stderr and stops its child process group if interrupted. An interrupted
epoch or LFCC batch can be repeated; committed checkpoints remain usable.
A writer lock also rejects a concurrent workflow. Large outputs are gitignored.

References: [random oversampling](https://imbalanced-learn.org/stable/over_sampling.html),
[Swin paper](https://arxiv.org/abs/2103.14030),
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
