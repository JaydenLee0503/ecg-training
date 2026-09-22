#!/usr/bin/env python
"""Patient-grouped comparison of VMD descriptors and standard WST with the same VQC.

Uses ECGData.mat exclusively for training. Source matching lives in
verify_ecg_sources.py; source excerpts never enter either feature matrix.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/matched_vqc.py \
        --vmd-features features/fixed_K8_a2000_L500_it2000_comparison.npz

Each fold is checkpointed and can be resumed under the identical manifest. The
outer test folds are evaluated only after fitting the fixed 40-epoch model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ecgvmd as E

ARMS = ('VMD descriptors', 'WST log')


def digest_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def checked_splits(X, y, patients, n_folds=5, seed=0):
    splits = list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
                  .split(X, y, patients))
    coverage = np.zeros(len(y), dtype=int)
    for tr, te in splits:
        if set(patients[tr]) & set(patients[te]):
            raise ValueError('A patient crosses a training/test boundary')
        if set(y[tr]) != set(E.CLASS_ORDER) or set(y[te]) != set(E.CLASS_ORDER):
            raise ValueError('Every training and test fold must contain every class')
        coverage[te] += 1
    if not np.all(coverage == 1):
        raise ValueError('Every window must have exactly one outer test prediction')
    return splits


def prepare(args):
    ds = E.load_ecgdata(args.mat)
    f = np.load(args.vmd_features, allow_pickle=True)
    cfg = E.Config(**json.loads(str(f['__config'])))
    if cfg.max_iter < 2000 or cfg.seg_mode != 'fixed':
        raise ValueError('Rebuild fixed-window VMD features with --max-iter 2000 or higher')
    W, y, row_ids = E.segment(ds, cfg, grouping='row')
    patients, sources = ds.patient_ids[row_ids], ds.source_ids[row_ids]
    if not np.array_equal(y.astype(str), f['__y'].astype(str)):
        raise ValueError('VMD feature labels do not match the segmented data')
    if not (np.array_equal(f['__groups'], row_ids) or np.array_equal(f['__groups'], patients)):
        raise ValueError('VMD feature row groups do not match this dataset/configuration')

    # Only the 28 descriptors per VMD mode, without rhythm or global extras.
    wanted = [f'u{k+1}_{feat}' for feat in E.MODE_FEATS for k in range(cfg.K)]
    saved_names = list(f['names::VMD modes'])
    idx = [saved_names.index(n) for n in wanted]
    Xv = f['X::VMD modes'][:, idx].copy()
    if len(Xv) != len(W):
        raise ValueError('VMD feature window count differs')
    if '__imfs' not in f or '__iters' not in f:
        raise ValueError('VMD archive needs --keep-imfs for alignment verification and iteration counts')

    # Recompute several rows from each class to catch stale/misaligned cache data.
    probes = np.concatenate([np.flatnonzero(y == c)[[0, 19, -1]] for c in E.CLASS_ORDER])
    vmd_kw = {k: getattr(cfg, k) for k in
              ('alpha', 'tau', 'K', 'dc', 'init', 'tol', 'fs')}
    check = E.vmd_batch(W[probes], max_iter=cfg.max_iter, **vmd_kw)
    if not np.allclose(check.modes.astype(np.float32), f['__imfs'][probes], atol=1e-7, rtol=0):
        raise ValueError('Recomputed VMD modes differ from the feature archive; rebuild it')
    descriptors, _ = E.mode_features(check.modes, check.omega_hz, cfg.fs, x=W[probes])
    if not np.allclose(descriptors, Xv[probes], atol=1e-6, rtol=1e-6):
        raise ValueError('Recomputed descriptors differ from the feature archive; rebuild it')

    iters = f['__iters'].copy()
    limits = np.full(len(W), cfg.max_iter)
    capped_before = int(np.count_nonzero(iters >= limits))
    ceiling = cfg.max_iter
    while np.any(iters >= limits) and ceiling < args.max_vmd_iter:
        remaining = np.flatnonzero(iters >= limits)
        ceiling = min(ceiling*2, args.max_vmd_iter)
        print(f'Retrying {len(remaining)} capped VMD windows at max_iter={ceiling}', flush=True)
        res = E.vmd_batch(W[remaining], max_iter=ceiling, **vmd_kw)
        Xv[remaining] = E.mode_features(res.modes, res.omega_hz, cfg.fs, x=W[remaining])[0]
        iters[remaining], limits[remaining] = res.iters, ceiling
    if np.any(iters >= limits):
        raise ValueError(f'{np.count_nonzero(iters >= limits)} VMD windows still capped; raise --max-vmd-iter')

    scat = E.scatter_batch(W, J=6, Q=(8, 1), T=64, max_order=2, fs=cfg.fs)
    Xw, wnames = E.scatter_features(scat, log=True)
    # Real-signal reference check for each class before starting long training.
    ref = E.scatter_kymatio(W[probes], J=6, Q=(8, 1), T=64, max_order=2, fs=cfg.fs)
    if not np.allclose(scat.coeffs[probes], ref.coeffs, atol=1e-12, rtol=0):
        raise ValueError('Scattering disagrees with Kymatio')
    blocks = dict(zip(ARMS, (Xv, Xw)))
    if not all(np.isfinite(x).all() for x in blocks.values()):
        raise ValueError('Non-finite features; investigate before training')
    meta = dict(config=cfg.to_dict(), windows_sha256=digest_array(W),
                subject_map_sha256=hashlib.sha256((ROOT/'ecgvmd/ecgdata_subjects.csv').read_bytes()).hexdigest(),
                feature_sha256={name: digest_array(x) for name, x in blocks.items()},
                vmd_initial_capped=capped_before, vmd_final_capped=0,
                vmd_max_observed_iterations=int(iters.max()),
                vmd_max_limit=int(limits.max()),
                wst=dict(J=6,Q=[8,1],T=64,max_order=2,log=True,reduce=None),
                n_windows=len(y), n_rows=len(np.unique(row_ids)),
                n_sources=len(np.unique(sources)), n_patients=len(np.unique(patients)),
                features={name:x.shape[1] for name,x in blocks.items()})
    return blocks, y.astype(str), patients, sources, row_ids, iters, limits, wanted, wnames, meta


def fit_one(arm, seed, fold, X, y, tr, te, feature_names, args):
    stem = f'{"vmd" if arm == ARMS[0] else "wst"}_seed{seed}_fold{fold}'
    checkpoint = args.out / f'{stem}.npz'
    if checkpoint.exists():
        z = np.load(checkpoint, allow_pickle=False)
        if not np.array_equal(z['test_indices'], te):
            raise ValueError(f'Checkpoint test rows differ: {checkpoint}')
        return arm, seed, fold, z['predictions'].astype(str), float(z['seconds'])
    started = time.time()
    print(f'START {arm} seed={seed} fold={fold}, train={len(tr)} test={len(te)}', flush=True)
    model = make_pipeline(E.MRMRSelector(k=args.k), E.TanhAngleScaler(),
                          E.VQCClassifier(n_layers=args.layers, epochs=args.epochs,
                                          batch_size=args.batch_size, lr=args.lr,
                                          seed=seed, verbose=True))
    model.fit(X[tr], y[tr])
    if model[-1].n_qubits_ != args.k:
        raise ValueError('Feature selector returned the wrong qubit count')
    prediction = model.predict(X[te]).astype(str)
    elapsed = time.time()-started
    joblib.dump(model, args.out/f'{stem}.joblib', compress=3)
    model[-1].save(args.out/f'{stem}_weights.npz')
    with (args.out/f'{stem}.tmp').open('wb') as stream:
        np.savez_compressed(stream, test_indices=te, predictions=prediction, seconds=elapsed,
                            selected_indices=model[0].idx_,
                            selected_names=np.asarray(feature_names)[model[0].idx_],
                            train_indices=tr)
    (args.out/f'{stem}.tmp').replace(checkpoint)
    print(f'DONE {arm} seed={seed} fold={fold}: {elapsed:.0f}s', flush=True)
    return arm, seed, fold, prediction, elapsed


def summarize(out, collected, y, patients, sources, row_ids, fold_of, seeds):
    rows, predictions = [], {}
    for arm in ARMS:
        for seed in range(seeds):
            parts = [collected.get((arm, seed, fold)) for fold in range(5)]
            if any(part is None for part in parts):
                continue
            pred = np.empty(len(y), dtype='<U3')
            for fold, (p, elapsed) in enumerate(parts):
                pred[fold_of == fold] = p
            name = f'{arm} + VQC seed={seed}'
            row = E.metrics_row(name, y, pred, patients, arm=arm, seed=seed,
                                seconds=sum(p[1] for p in parts))
            # Generic E.metrics_row calls aggregation "record"; this run votes by patient.
            row = {k.replace('record_', 'patient_'):v for k,v in row.items()}
            rows.append(row)
            predictions[name] = pred
    if not rows:
        return
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    np.savez_compressed(out/'predictions.npz', __y=y, __groups=patients,
                        __sources=sources, __row_ids=row_ids, __fold=fold_of, **predictions)
    pd.DataFrame(rows).groupby('arm')[['accuracy','macro_f1','patient_accuracy','patient_macro_f1']].agg(
        ['mean','std']).to_csv(out/'summary.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mat', default=None)
    ap.add_argument('--vmd-features', required=True)
    ap.add_argument('--out', type=Path, default=ROOT/'results/patient_vmd_wst_vqc')
    ap.add_argument('--max-vmd-iter', type=int, default=32000)
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=.05)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--n-jobs', type=int, default=5)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    blocks,y,patients,sources,row_ids,iters,limits,vnames,wnames,meta = prepare(args)
    splits = checked_splits(blocks[ARMS[0]], y, patients)
    fold_of = np.full(len(y), -1)
    for fold, (_,te) in enumerate(splits):
        fold_of[te] = fold
    import pennylane, scipy, sklearn
    meta.update(protocol_version=1, k=args.k,layers=args.layers,epochs=args.epochs,
                batch_size=args.batch_size,lr=args.lr,seeds=list(range(args.seeds)),
                cv_seed=0, n_folds=5, fold_sha256=digest_array(fold_of),
                grouping='verified patient', scaler='TanhAngleScaler',selector='MRMRSelector',
                versions=dict(numpy=np.__version__,scipy=scipy.__version__,
                              sklearn=sklearn.__version__,pennylane=pennylane.__version__))
    manifest = args.out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != meta:
        raise ValueError('Output directory contains a different protocol; choose a new --out')
    manifest.write_text(json.dumps(meta, indent=2)+'\n')
    np.savez_compressed(args.out/'features.npz', vmd=blocks[ARMS[0]], wst=blocks[ARMS[1]],
                        y=y,patients=patients,sources=sources,row_ids=row_ids,fold=fold_of,
                        vmd_iters=iters,vmd_limits=limits,vmd_names=np.asarray(vnames),
                        wst_names=np.asarray(wnames))
    pd.DataFrame(dict(row=row_ids,patient=patients,source=sources,label=y,fold=fold_of)).to_csv(
        args.out/'folds.csv',index=False)
    print(json.dumps(meta, indent=2),flush=True)
    if args.prepare_only:
        return
    tasks = [(arm,seed,fold,tr,te) for seed in range(args.seeds)
             for fold,(tr,te) in enumerate(splits) for arm in ARMS]
    collected = {}
    results = Parallel(n_jobs=args.n_jobs, return_as='generator_unordered')(
        delayed(fit_one)(arm,seed,fold,blocks[arm],y,tr,te,
                         vnames if arm == ARMS[0] else wnames,args)
        for arm,seed,fold,tr,te in tasks)
    for arm,seed,fold,pred,elapsed in results:
        collected[(arm,seed,fold)] = (pred,elapsed)
        summarize(args.out,collected,y,patients,sources,row_ids,fold_of,args.seeds)
        print(f'Completed {len(collected)}/{len(tasks)} fold fits',flush=True)
    print(pd.read_csv(args.out/'metrics.csv')[['model','accuracy','macro_f1','patient_accuracy']].to_string(index=False))


if __name__ == '__main__':
    main()
