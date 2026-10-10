"""Resumable 30-second EMD/Hilbert ECGData experiment."""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import argparse
import sys
import time
import traceback

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROTOCOL = ROOT / 'ecgdata_eyeball_protocol_v3.json'
DEFAULT_RUN = ROOT / 'results/ecgdata_eyeball_30s_v3'
DEPS = REPO / 'experiments/acs_omi_vmd_wst_vqc/results/eyeball_dependencies_v1'
sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(REPO))
from ecgvmd.rotational import extract, NAMES
from .ecgdata import (atomic_json, atomic_npz, file_hash, read_json, safe_output,
                      array_hash, writer_lock)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _record_event(run, status, **info):
    with (run / 'attempts.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                     status=status, **info), allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _sources():
    paths = [PROTOCOL, Path(__file__), REPO/'ecgvmd/rotational.py',
             REPO/'ecgvmd/data.py', REPO/'ecgvmd/ecgdata_subjects.csv',
             REPO/'attention method/ecgdata.py',
             REPO/'attention method/attention_ecg/ecgdata.py']
    return {str(p.relative_to(REPO)): file_hash(p) for p in paths}


def _load_inputs(mat_path):
    from ecgvmd.data import load_ecgdata
    protocol = read_json(PROTOCOL)
    mat_path = Path(mat_path).resolve()
    ds = load_ecgdata(str(mat_path))
    cfg = protocol['data']['windows']
    length = cfg['samples']
    count = ds.n_samples // length
    if count < cfg['per_lead_row']:
        raise ValueError('Not enough complete 30-second clips in a source row')

    from .ecgdata import DEFAULT_CACHE, verify_cache
    _, baseline = verify_cache(DEFAULT_CACHE)
    if len(np.unique(baseline['row_ids'])) != len(ds.record_ids):
        raise ValueError('ECGData row count differs from the corrected patient reference')
    for row in ds.record_ids:
        where = baseline['row_ids'] == row
        if (not np.any(where) or len(np.unique(baseline['patients'][where])) != 1 or
                str(baseline['patients'][where][0]) != str(ds.patient_ids[row]) or
                len(np.unique(baseline['y'][where])) != 1 or
                int(baseline['y'][where][0]) != protocol['data']['classes'].index(str(ds.labels[row]))):
            raise ValueError(f'ECGData row {row} differs from corrected patient reference')
    labels = np.array([protocol['data']['classes'].index(str(x)) for x in ds.labels], dtype=np.int64)
    row_folds = np.empty(ds.n_records, dtype=np.int64)
    for row in ds.record_ids:
        row_folds[row] = np.unique(baseline['fold'][baseline['row_ids'] == row])[0]
    rng = np.random.default_rng(cfg['selection_seed'])
    starts = []
    for _ in ds.record_ids:
        selected = np.sort(rng.choice(count, cfg['per_lead_row'], replace=False))
        starts.extend((selected * length).tolist())
    starts = np.asarray(starts, dtype=np.int64)
    row_ids = np.repeat(ds.record_ids, cfg['per_lead_row'])
    y = np.repeat(labels, cfg['per_lead_row'])
    patients = np.repeat(ds.patient_ids.astype(str), cfg['per_lead_row'])
    sources = np.repeat(ds.source_ids.astype(str), cfg['per_lead_row'])
    folds = np.repeat(row_folds, cfg['per_lead_row'])
    ids = np.arange(len(starts), dtype=np.int64)
    signals = np.empty((len(starts), length), dtype=np.float64)
    for i, (row, start) in enumerate(zip(row_ids, starts)):
        w = np.asarray(ds.data[row, start:start+length], dtype=np.float64)
        sd = float(w.std())
        if not np.isfinite(w).all() or not np.isfinite(sd) or sd <= 0:
            raise ValueError(f'Invalid ECGData clip at row={row}, start={start}')
        signals[i] = (w - w.mean()) / sd
    if (len(ids) != protocol['data']['cohort_expectation']['windows'] or
            len(np.unique(patients)) != protocol['data']['cohort_expectation']['patients']):
        raise ValueError('Unexpected cohort size')
    for p in np.unique(patients):
        i = patients == p
        if len(np.unique(folds[i])) != 1 or len(np.unique(y[i])) != 1:
            raise ValueError('Patient crosses folds or labels')
    data = dict(signals=signals, ids=ids, row_ids=row_ids, starts=starts, y=y,
                patients=patients, sources=sources, fold=folds)
    return mat_path, protocol, data


def _context(mat_path, protocol, data):
    return dict(protocol=protocol, sources=_sources(), mat_sha256=file_hash(mat_path),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__),
                cohort=dict(windows=len(data['ids']), patients=len(np.unique(data['patients'])),
                            signals_sha256=array_hash(data['signals']),
                            patient_sha256=array_hash(data['patients']),
                            fold_sha256=array_hash(data['fold']),
                            starts_sha256=array_hash(data['starts'])))


def _artifact(run, i):
    return run/'features'/f'clip_{i:04d}.npz'


def _verified_clip(path, context):
    marker = path.with_suffix('.json')
    if not path.exists() and not marker.exists():
        return None
    if not path.exists() or not marker.exists():
        raise ValueError(f'Orphaned clip artifact: {path}')
    meta = read_json(marker)
    if meta.get('context_sha256') != context or meta.get('sha256') != file_hash(path):
        raise ValueError(f'Clip identity/checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as z:
        status = str(z['status'].item())
        if status == 'complete':
            x = z['features']
            if x.shape != (12,) or not np.isfinite(x).all():
                raise ValueError(f'Invalid feature vector in {path}')
            return dict(status=status, features=x, details=json.loads(str(z['details'].item())))
        return dict(status=status, error=str(z['error'].item()),
                    details=json.loads(str(z['details'].item())))


def _extract_clip(signal, fs, spec):
    started = time.perf_counter()
    try:
        features, details, _ = extract(signal, fs, spec, retain=False)
        return dict(status='complete', features=features, details=details,
                    seconds=time.perf_counter()-started)
    except Exception as exc:
        return dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                    details=getattr(exc, 'diagnostics', {}), seconds=time.perf_counter()-started)


def extract_all(mat=REPO/'ECGData.mat', run=DEFAULT_RUN, *, execute=False):
    if not execute:
        raise ValueError('Extraction requires explicit --execute-extraction')
    run = safe_output(run, 'results')
    mat_path, protocol, data = _load_inputs(mat)
    context = _context(mat_path, protocol, data)
    context_sha = hashlib.sha256(_canonical(context).encode()).hexdigest()
    with writer_lock(run):
        run.mkdir(parents=True, exist_ok=True)
        mp = run/'manifest.json'
        if mp.exists():
            if read_json(mp) != context:
                raise ValueError('Run context changed; use a new output folder')
            if (run/'manifest.sha256').read_text().strip() != file_hash(mp):
                raise ValueError('Run manifest checksum mismatch')
        else:
            atomic_json(mp, context)
            (run/'manifest.sha256').write_text(file_hash(mp)+'\n')
        (run/'features').mkdir(exist_ok=True)
        (run/'status.json').write_text(json.dumps(dict(status='running', stage='extraction',
            context_sha256=context_sha, total=len(data['ids']), updated_utc=time.time()), indent=2)+'\n')
        _record_event(run, 'extraction_started', clips=len(data['ids']), context_sha256=context_sha)
        failures = []
        pending = []
        for i in range(len(data['ids'])):
            saved = _verified_clip(_artifact(run, i), context_sha)
            if saved is None:
                pending.append(i)
            elif saved['status'] == 'failed':
                failures.append(dict(clip=int(i), row_id=int(data['row_ids'][i]),
                    patient_id=str(data['patients'][i]), y=int(data['y'][i]),
                    fold=int(data['fold'][i]), error=saved['error'], details=saved['details']))

        def save_result(i, result):
            path = _artifact(run, i)
            payload = dict(status=np.array(result['status']),
                details=np.array(json.dumps(result['details'], allow_nan=False)))
            if result['status'] == 'complete':
                payload['features'] = result['features']
            else:
                payload['error'] = np.array(result['error'])
                failures.append(dict(clip=int(i), row_id=int(data['row_ids'][i]),
                    patient_id=str(data['patients'][i]), y=int(data['y'][i]),
                    fold=int(data['fold'][i]), error=result['error'], details=result['details']))
            atomic_npz(path, **payload)
            atomic_json(path.with_suffix('.json'), dict(context_sha256=context_sha,
                sha256=file_hash(path), seconds=result['seconds'], record_id=int(data['row_ids'][i]),
                patient_id=str(data['patients'][i]), start_sample=int(data['starts'][i]),
                status=result['status']))
            _record_event(run, result['status'], clip=int(i), record_id=int(data['row_ids'][i]),
                patient_id=str(data['patients'][i]), seconds=result['seconds'], error=result.get('error'))

        spec = _feature_spec(protocol)
        with ProcessPoolExecutor(max_workers=protocol['execution']['extraction_workers']) as pool:
            for first in range(0, len(pending), 8):
                batch = pending[first:first+8]
                futures = {pool.submit(_extract_clip, data['signals'][i],
                    protocol['data']['sampling_rate_hz'], spec): i for i in batch}
                for future in as_completed(futures):
                    save_result(futures[future], future.result())
                finished = len(data['ids']) - len(pending) + min(first+len(batch), len(pending))
                successes = finished - len(failures)
                atomic_json(run/'status.json', dict(status='running', stage='extraction',
                    attempted=finished, total=len(data['ids']), successful=successes,
                    failed=len(failures), workers=protocol['execution']['extraction_workers'],
                    updated_utc=time.time()))
                print(f'extraction {finished}/{len(data["ids"])}; success={successes}; failed={len(failures)}', flush=True)
        summary = dict(status='complete', attempted=len(data['ids']),
                       successful=len(data['ids'])-len(failures), failed=len(failures),
                       coverage=(len(data['ids'])-len(failures))/len(data['ids']),
                       class_failures={str(c):sum(x['y']==c for x in failures) for c in range(3)},
                       folds=len(np.unique(data['fold'])), finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        atomic_json(run/'extraction_summary.json', summary)
        atomic_json(run/'failures.json', failures)
        atomic_npz(run/'cohort.npz', **{k:v for k,v in data.items() if k!='signals'})
        atomic_json(run/'status.json', dict(**summary, stage='extraction'))
        _record_event(run, 'extraction_complete', result=summary)
    return summary


def _feature_spec(protocol):
    spec = protocol['features']
    return dict(minimum_seconds=protocol['data']['windows']['seconds'],
        emd=dict(version=spec['emd']['version'], components=spec['components'],
            parameters={k:v for k,v in spec['emd'].items()
                if k in ('spline_kind','nbsym','extrema_detection','MAX_ITERATION','FIXE','FIXE_H',
                         'energy_ratio_thr','std_thr','svar_thr','total_power_thr','range_thr','scale_factor')},
            reconstruction_atol_mV=spec['reconstruction_atol'],
            reconstruction_rtol=spec['reconstruction_rtol']),
        analytic=dict(trim_seconds=spec['analytic']['trim_seconds_each_end'],
            envelope_floor_mV=spec['analytic']['envelope_floor_mV'],
            envelope_floor_relative=spec['analytic']['envelope_floor_relative'],
            minimum_valid_phase_fraction=spec['analytic']['minimum_valid_phase_fraction']))


def train_models(run=DEFAULT_RUN, *, execute=False):
    if not execute:
        raise ValueError('Model fitting requires explicit --execute-training')
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    import joblib
    run = safe_output(run, 'results')
    protocol = read_json(PROTOCOL)
    summary = read_json(run/'extraction_summary.json')
    if (summary['status'] != 'complete' or
            summary['coverage'] < protocol['execution']['minimum_extraction_coverage_for_fitting']):
        raise ValueError('Extraction coverage below the frozen 90% model-fitting gate')
    cohort = {k:v for k,v in np.load(run/'cohort.npz', allow_pickle=False).items()}
    X = np.full((len(cohort['ids']), 12), np.nan)
    for i in cohort['ids']:
        saved = _verified_clip(_artifact(run, int(i)), hashlib.sha256(_canonical(read_json(run/'manifest.json')).encode()).hexdigest())
        if saved and saved['status'] == 'complete':
            X[i] = saved['features']
    valid = np.isfinite(X).all(1)
    fold_results = []
    run.mkdir(parents=True, exist_ok=True)
    _record_event(run, 'training_started', valid_clips=int(valid.sum()), coverage=float(valid.mean()))
    for fold in range(5):
        tr = np.flatnonzero((cohort['fold'] != fold) & valid)
        te = np.flatnonzero((cohort['fold'] == fold) & valid)
        if set(cohort['y'][tr]) != {0,1,2} or set(cohort['y'][te]) != {0,1,2}:
            raise ValueError(f'Fold {fold} lacks a class after extraction failures; no fitting performed for this fold')
        for spec in protocol['models']:
            name = spec['name']
            path = run/f'models/{name}_fold{fold}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if (not np.array_equal(saved['indices'], te) or
                            saved['probabilities'].shape != (len(te),3) or
                            not np.array_equal(saved['y'],cohort['y'][te]) or
                            not np.array_equal(saved['patients'],cohort['patients'][te]) or
                            not np.array_equal(saved['train_indices'],tr)):
                        raise ValueError(f'Existing fold output mismatch: {path}')
                continue
            if name == 'balanced_logistic':
                model = make_pipeline(StandardScaler(), LogisticRegression(C=spec['C'],
                    class_weight=spec['class_weight'], solver=spec['solver'],
                    max_iter=spec['max_iter'], random_state=spec['random_state']))
            else:
                model = make_pipeline(StandardScaler(), SVC(C=spec['C'], gamma=spec['gamma'],
                    class_weight=spec['class_weight'], probability=spec['probability'],
                    random_state=spec['random_state']))
            started = time.perf_counter()
            model.fit(X[tr], cohort['y'][tr])
            probabilities = model.predict_proba(X[te])
            if not np.isfinite(probabilities).all():
                raise ValueError('Nonfinite model scores')
            model_path = path.with_suffix('.joblib')
            tmp = model_path.with_suffix('.joblib.tmp')
            joblib.dump(model, tmp)
            os.replace(tmp, model_path)
            atomic_npz(path, indices=te, y=cohort['y'][te], patients=cohort['patients'][te],
                       probabilities=probabilities, pred=probabilities.argmax(1),
                       train_indices=tr, seconds=np.array(time.perf_counter()-started))
            _record_event(run, 'model_fold_complete', model=name, fold=fold,
                          train_windows=len(tr), test_windows=len(te), seconds=time.perf_counter()-started)
        fold_results.append(fold)
    atomic_json(run/'training_summary.json', dict(status='complete', folds=fold_results,
                models=[x['name'] for x in protocol['models']], valid_windows=int(valid.sum()),
                training_completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    return read_json(run/'training_summary.json')


def report(run=DEFAULT_RUN):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, average_precision_score
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    run = safe_output(run, 'results')
    protocol = read_json(PROTOCOL)
    summary = read_json(run/'extraction_summary.json')
    cohort = {k:v for k,v in np.load(run/'cohort.npz', allow_pickle=False).items()}
    ys, patients = cohort['y'], cohort['patients'].astype(str)
    valid = np.zeros(len(ys), dtype=bool)
    arms, uncertainty, saved_predictions = {}, {}, {}
    for spec in protocol['models']:
        pred = np.full(len(ys), -1, dtype=int)
        probs = np.full((len(ys),3), np.nan)
        coverage = np.zeros(len(ys), dtype=int)
        for fold in range(5):
            with np.load(run/f'models/{spec["name"]}_fold{fold}.npz', allow_pickle=False) as z:
                idx = z['indices']
                pred[idx], probs[idx] = z['pred'], z['probabilities']
                coverage[idx] += 1
        valid = coverage == 1
        if not valid.any() or not np.isfinite(probs[valid]).all():
            raise ValueError(f'Invalid out-of-fold coverage for {spec["name"]}')
        patient_y, patient_pred, patient_probs = [], [], []
        for patient in np.unique(patients[valid]):
            idx = (patients == patient) & valid
            labels = np.unique(ys[idx])
            if len(labels) != 1:
                raise ValueError('Patient label inconsistency')
            patient_y.append(labels[0])
            patient_probs.append(probs[idx].mean(0))
            patient_pred.append(np.mean(probs[idx], axis=0).argmax())
        patient_probs = np.asarray(patient_probs)
        yv, pv = ys[valid], probs[valid]
        metrics = dict(window_accuracy=float(accuracy_score(yv,pred[valid])),
            window_macro_f1=float(f1_score(yv,pred[valid],average='macro')),
            window_balanced_accuracy=float(balanced_accuracy_score(yv,pred[valid])),
            window_average_precision_ovr=float(average_precision_score(yv,pv,average='macro')),
            patient_vote_accuracy=float(accuracy_score(patient_y,patient_pred)),
            patient_vote_macro_f1=float(f1_score(patient_y,patient_pred,average='macro')),
            patient_vote_average_precision_ovr=float(average_precision_score(patient_y,patient_probs,average='macro')),
            valid_windows=int(valid.sum()), total_windows=len(ys), patient_count=len(patient_y),
            class_coverage={str(c):int(np.sum(ys[valid]==c)) for c in range(3)})
        precision, recall, class_f1, class_support = precision_recall_fscore_support(
            yv, pred[valid], labels=[0,1,2], zero_division=0)
        metrics['window_confusion_matrix'] = confusion_matrix(yv,pred[valid],labels=[0,1,2]).tolist()
        metrics['window_per_class'] = {protocol['data']['classes'][c]:dict(
            precision=float(precision[c]), recall=float(recall[c]), f1=float(class_f1[c]),
            support=int(class_support[c])) for c in range(3)}
        metrics['per_fold'] = {}
        for fold in range(5):
            idx = (cohort['fold'] == fold) & valid
            metrics['per_fold'][str(fold)] = dict(n=int(idx.sum()),
                accuracy=float(accuracy_score(ys[idx],pred[idx])),
                macro_f1=float(f1_score(ys[idx],pred[idx],average='macro')),
                balanced_accuracy=float(balanced_accuracy_score(ys[idx],pred[idx])),
                confusion_matrix=confusion_matrix(ys[idx],pred[idx],labels=[0,1,2]).tolist())
        patient_pred_array = np.asarray(patient_pred)
        metrics['patient_vote_confusion_matrix'] = confusion_matrix(
            patient_y,patient_pred_array,labels=[0,1,2]).tolist()
        metrics['patients_with_any_failure'] = int(sum(
            np.any((patients == patient) & ~valid) for patient in np.unique(patients)))
        metrics['patients_with_no_valid_clip'] = int(sum(
            not np.any((patients == patient) & valid) for patient in np.unique(patients)))
        arms[spec['name']] = metrics
        saved_predictions[spec['name']] = (pred, probs)
        atomic_npz(run/f'{spec["name"]}_predictions.npz', indices=np.flatnonzero(valid),
                   y=ys[valid], patients=patients[valid], pred=pred[valid], probabilities=probs[valid])
        unique_patients = np.unique(patients[valid])
        patient_rows = []
        for patient in unique_patients:
            idx = (patients == patient) & valid
            patient_rows.append((int(np.unique(ys[idx])[0]), probs[idx].mean(0)))
        patient_y = np.asarray([x[0] for x in patient_rows])
        patient_prob = np.asarray([x[1] for x in patient_rows])
        rng = np.random.default_rng(protocol['evaluation']['bootstrap_seed'])
        draws = {key:[] for key in ('window_macro_f1','window_accuracy','window_balanced_accuracy',
                                     'window_average_precision_ovr','patient_vote_accuracy',
                                     'patient_vote_macro_f1','patient_vote_average_precision_ovr')}
        for _ in range(protocol['evaluation']['bootstrap_resamples']):
            multiplicity = np.bincount(rng.integers(len(unique_patients), size=len(unique_patients)),
                                       minlength=len(unique_patients))
            weight = np.zeros(len(ys), dtype=np.float64)
            patient_weight = np.zeros(len(unique_patients), dtype=np.float64)
            for j, patient in enumerate(unique_patients):
                weight[(patients == patient) & valid] = multiplicity[j]
                patient_weight[j] = multiplicity[j]
            keep = valid & (weight > 0)
            if not np.any(keep):
                continue
            draws['window_macro_f1'].append(f1_score(ys[keep],pred[keep],average='macro',sample_weight=weight[keep]))
            draws['window_accuracy'].append(accuracy_score(ys[keep],pred[keep],sample_weight=weight[keep]))
            draws['window_balanced_accuracy'].append(balanced_accuracy_score(ys[keep],pred[keep],sample_weight=weight[keep]))
            draws['window_average_precision_ovr'].append(average_precision_score(ys[keep],probs[keep],
                average='macro',sample_weight=weight[keep]))
            pkeep = patient_weight > 0
            p_pred = patient_prob.argmax(1)
            draws['patient_vote_accuracy'].append(accuracy_score(patient_y[pkeep],p_pred[pkeep],sample_weight=patient_weight[pkeep]))
            draws['patient_vote_macro_f1'].append(f1_score(patient_y[pkeep],p_pred[pkeep],average='macro',sample_weight=patient_weight[pkeep]))
            draws['patient_vote_average_precision_ovr'].append(average_precision_score(patient_y[pkeep],patient_prob[pkeep],
                average='macro',sample_weight=patient_weight[pkeep]))
        uncertainty[spec['name']] = {key:np.quantile(values,[.025,.975]).tolist()
                                      for key,values in draws.items()}
    output = dict(status='complete', protocol_id=protocol['id'], extraction=summary,
                  models=arms, classes=protocol['data']['classes'],
                  training=dict(settings=protocol['models'],
                      fold_fit_seconds=[row for row in map(json.loads,(run/'attempts.jsonl').read_text().splitlines())
                                        if row.get('status')=='model_fold_complete'],
                      source_hashes=read_json(run/'manifest.json')['sources'],
                      environment=read_json(run/'manifest.json')['versions']),
                  uncertainty_95=uncertainty, bootstrap=dict(unit='patient',
                      resamples=protocol['evaluation']['bootstrap_resamples'],
                      seed=protocol['evaluation']['bootstrap_seed'],
                      scope=protocol['evaluation']['interval_scope']),
                  limitations=protocol['limitations'], predictions_sha256={
                      name:file_hash(run/f'{name}_predictions.npz') for name in saved_predictions})
    atomic_json(run/'report.json', output)
    return output


def audit_run(run=DEFAULT_RUN):
    """Verify all extracted clips and reloaded out-of-fold model probabilities."""
    import joblib
    run = safe_output(run, 'results')
    protocol = read_json(PROTOCOL)
    manifest = read_json(run/'manifest.json')
    manifest_sha = file_hash(run/'manifest.json')
    if (run/'manifest.sha256').read_text().strip() != manifest_sha:
        raise ValueError('Extraction manifest checksum mismatch')
    context_sha = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    with np.load(run/'cohort.npz', allow_pickle=False) as z:
        cohort = {k:z[k] for k in z.files}
    if len(cohort['ids']) != 1620 or len(np.unique(cohort['patients'])) != 80:
        raise ValueError('Unexpected saved cohort')
    X = np.full((len(cohort['ids']),12), np.nan)
    outcomes = []
    for i in cohort['ids']:
        saved = _verified_clip(_artifact(run,int(i)), context_sha)
        if saved is None:
            raise ValueError(f'Missing clip {i}')
        outcomes.append(saved['status'])
        if saved['status'] == 'complete':
            X[i] = saved['features']
    valid = np.isfinite(X).all(1)
    failures = read_json(run/'failures.json')
    summary = read_json(run/'extraction_summary.json')
    if (int(valid.sum()) != summary['successful'] or int((~valid).sum()) != summary['failed'] or
            len(failures) != summary['failed']):
        raise ValueError('Extraction summary does not match clip artifacts')
    model_checks = []
    for model_spec in protocol['models']:
        name = model_spec['name']
        for fold in range(5):
            expected = np.flatnonzero((cohort['fold']==fold)&valid)
            path = run/f'models/{name}_fold{fold}.npz'
            with np.load(path,allow_pickle=False) as z:
                if (not np.array_equal(z['indices'],expected) or
                    not np.array_equal(z['y'],cohort['y'][expected]) or
                    not np.array_equal(z['patients'],cohort['patients'][expected]) or
                    z['probabilities'].shape != (len(expected),3) or
                    not np.isfinite(z['probabilities']).all()):
                    raise ValueError(f'Invalid prediction artifact: {path}')
                stored = z['probabilities'].copy()
            model = joblib.load(path.with_suffix('.joblib'))
            actual = model.predict_proba(X[expected])
            error = float(np.max(np.abs(actual-stored)))
            if not np.array_equal(actual.argmax(1),stored.argmax(1)) or error > 1e-12:
                raise ValueError(f'Reloaded predictions differ: {path} max_abs={error}')
            model_checks.append(dict(model=name,fold=fold,n=len(expected),
                                     max_probability_difference=error))
    inventory = {}
    for path in sorted(run.rglob('*')):
        if not path.is_file() or path.name in ('artifact_inventory.json','completed.json') or path.name == '.writer.lock':
            continue
        inventory[str(path.relative_to(run))] = dict(bytes=path.stat().st_size,sha256=file_hash(path))
    atomic_json(run/'artifact_inventory.json',dict(status='complete',files=inventory,
                files_checked=len(inventory),successful_clips=int(valid.sum()),
                failed_clips=int((~valid).sum()),model_checks=model_checks))
    result = dict(status='complete',protocol_id=protocol['id'],
        context_sha256=context_sha,artifact_inventory_sha256=file_hash(run/'artifact_inventory.json'),
        files_checked=len(inventory),successful_clips=int(valid.sum()),failed_clips=int((~valid).sum()),
        coverage=float(valid.mean()),reloaded_model_fits=len(model_checks),
        maximum_probability_difference=max(x['max_probability_difference'] for x in model_checks),
        verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(run/'completed.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--mat', type=Path, default=REPO/'ECGData.mat')
    p.add_argument('--out', type=Path, default=DEFAULT_RUN)
    p.add_argument('--execute-extraction', action='store_true', required=True)
    p = sub.add_parser('train')
    p.add_argument('--out', type=Path, default=DEFAULT_RUN)
    p.add_argument('--execute-training', action='store_true', required=True)
    p = sub.add_parser('report')
    p.add_argument('--out', type=Path, default=DEFAULT_RUN)
    p = sub.add_parser('audit')
    p.add_argument('--out', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.action == 'extract':
        result = extract_all(args.mat, args.out, execute=args.execute_extraction)
    elif args.action == 'train':
        result = train_models(args.out, execute=args.execute_training)
    elif args.action == 'report':
        result = report(args.out)
    else:
        result = audit_run(args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
