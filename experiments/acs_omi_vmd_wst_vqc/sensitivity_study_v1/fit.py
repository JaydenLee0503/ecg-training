"""New balanced fits; calibration cutoffs frozen before validation inference."""
import copy
import time
import traceback
import warnings

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from . import common as c
from .metrics import Curve, at_threshold
from experiments.acs_omi_vmd_wst_vqc.pipeline import make_classifiers
from experiments.acs_omi_vmd_wst_vqc.training import CheckpointVQC, atomic_joblib


def save_predictions(path, data, indices, score, thresholds=None):
    score = np.asarray(score)
    if score.shape != (len(indices),) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError('Invalid probabilities')
    values = dict(score=score, prediction=(score >= .5).astype(int), y=data['y'][indices],
                  record_ids=data['record_ids'][indices], patients=data['patients'][indices])
    if thresholds is not None:
        values['thresholds'] = np.asarray(thresholds)
        values['target_predictions'] = score[:, None] >= np.asarray(thresholds)[None, :]
    c.npz(path, **values)


def freeze_thresholds(folder, y, score, targets, ctx):
    points = Curve(y, score).operating_points(targets)
    value = dict(context_sha=ctx, targets_percent=targets, selected_on='calibration patients only',
                 calibration_points=points, validation_labels_used=False)
    path = folder / 'thresholds.json'
    if path.exists() and c.read(path) != value:
        raise ValueError('Previously frozen thresholds differ')
    c.write(path, value)
    return [p['threshold'] for p in points]


def run_trial(arm, name, device='cpu'):
    manifest, ctx = c.manifest()
    out = c.RUN / 'models' / arm / name
    with c.lock(out), threadpool_limits(limits=1):
        if c.verified(out, ctx):
            print(f'{arm}/{name}: verified complete', flush=True)
            return
        started = time.perf_counter()
        c.write(out / 'status.json', dict(status='running', started_at_utc=c.now(), arm=arm, model=name))
        try:
            prepared = c.RUN / 'prepared'
            # Parent checks all prepared hashes; each worker pins the marker and its actual inputs.
            marker = c.read(prepared / 'completed.json')
            if marker['context_sha'] != ctx:
                raise ValueError('Prepared context differs')
            needed = ['cohort.npz', 'lfcc_normalized.npy' if name.startswith('swin') else f'{arm}.npz']
            for filename in needed:
                if c.sha(prepared / filename) != marker['files'][filename]:
                    raise ValueError('Changed prepared input')
            data = c.arrays(prepared / 'cohort.npz')
            tr = data['sampled_indices']
            cal = np.flatnonzero(data['partitions'] == 'calibration')
            val = np.flatnonzero(data['partitions'] == 'validation')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if name.startswith('swin'):
                    legacy = c.legacy_swin()
                    environment = legacy.setup(device)
                    env_path = out / 'environment.json'
                    if env_path.exists() and c.read(env_path) != environment:
                        raise ValueError('Changed Swin environment on resume')
                    c.write(env_path, environment)
                    X = np.load(prepared / 'lfcc_normalized.npy', mmap_mode='c')
                    spec = manifest['balanced_settings']
                    seed = int(name[-1])
                    # The frozen helper's `val` argument receives calibration indices only.
                    legacy.fit_swin(out / 'fit', data, X, tr, cal, seed, spec, device, ctx)
                    calibration = c.arrays(out / 'fit/predictions.npz')
                    np.testing.assert_array_equal(calibration['record_ids'], data['record_ids'][cal])
                    score = calibration['score']
                    thresholds = freeze_thresholds(out, data['y'][cal], score,
                                                   manifest['protocol']['targets_percent'], ctx)
                    save_predictions(out / 'calibration_predictions.npz', data, cal, score, thresholds)
                    from attention_ecg.model import CepstralSwin
                    model = CepstralSwin(**spec['model']).to(device)
                    model.load_state_dict(legacy.load_checkpoint(out / 'fit', ctx)['model'])
                    validation = legacy.predict(model, X, val, spec['training']['batch_size'], device)
                    restored = CepstralSwin(**spec['model']).to(device)
                    restored.load_state_dict(legacy.load_checkpoint(out / 'fit', ctx)['model'])
                    np.testing.assert_array_equal(validation,
                        legacy.predict(restored, X, val, spec['training']['batch_size'], device))
                    settings = dict(model=spec['model'], training=spec['training'], seed=seed)
                else:
                    X = c.arrays(prepared / f'{arm}.npz')['X']
                    protocol = copy.deepcopy(manifest['original_settings'])
                    protocol['vqc']['class_weight'] = None
                    protocol['controls']['logistic']['class_weight'] = None
                    if name.startswith('vqc'):
                        params = {k: v for k, v in protocol['vqc'].items() if k not in ('class', 'seeds')}
                        model = CheckpointVQC(seed=int(name[-1]), **params)
                        model.fit(X[tr], data['y'][tr], out / 'epochs', ctx)
                        c.write(out / 'history.json', model.history_)
                        model.save_weights(out / 'weights.npz')
                    else:
                        if arm == 'lfcc':
                            spec = manifest['balanced_settings']['control']
                            params = {k: spec[k] for k in ('C', 'class_weight', 'max_iter', 'solver')}
                            model = LogisticRegression(**params, random_state=0)
                        else:
                            model = make_classifiers(protocol)[name]
                        model.fit(X[tr], data['y'][tr])
                    if list(model.classes_) != [0, 1]:
                        raise ValueError('Unexpected class ordering')
                    atomic_joblib(out / 'model.joblib', model)
                    restored = joblib.load(out / 'model.joblib')
                    score = model.predict_proba(X[cal])[:, 1]
                    np.testing.assert_array_equal(score, restored.predict_proba(X[cal])[:, 1])
                    thresholds = freeze_thresholds(out, data['y'][cal], score,
                                                   manifest['protocol']['targets_percent'], ctx)
                    save_predictions(out / 'calibration_predictions.npz', data, cal, score, thresholds)
                    validation = model.predict_proba(X[val])[:, 1]
                    np.testing.assert_array_equal(validation, restored.predict_proba(X[val])[:, 1])
                    unique = np.unique(tr)
                    save_predictions(out / 'training_predictions.npz', data, unique,
                                     model.predict_proba(X[unique])[:, 1])
                    settings = model.get_params()
                save_predictions(out / 'validation_predictions.npz', data, val, validation, thresholds)
            c.write(out / 'fit_record.json', dict(settings=settings, seconds=time.perf_counter()-started,
                warnings=[str(w.message) for w in caught], calibration_records=len(cal), validation_records=len(val),
                subfit_records=len(np.unique(tr)), balanced_training_examples=len(tr),
                calibration_reload_exact=True, validation_reload_exact=True,
                threshold_frozen_before_validation_inference=True, official_test_processed=False,
                completed_at_utc=c.now()))
            c.complete(out, ctx)
            c.write(out / 'status.json', dict(status='complete', completed_at_utc=c.now()))
            print(f'{arm}/{name}: complete in {time.perf_counter()-started:.1f}s', flush=True)
        except BaseException:
            failure = dict(status='failed', time_utc=c.now(), traceback=traceback.format_exc())
            c.write(out / f'failure_{time.time_ns()}.json', failure)
            c.write(out / 'status.json', failure)
            raise
