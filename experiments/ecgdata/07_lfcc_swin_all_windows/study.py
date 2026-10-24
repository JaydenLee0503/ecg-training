"""All-window ECGData LFCC/Swin study (ECG-07).

Reuses the v1 cepstral, model, fitting and metric code unchanged; only the window
set differs. Preparation and reporting never fit a model. Fitting needs an explicit
flag. Original v1 caches and runs are read, never written.
"""
import csv
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
ATTENTION = REPO/'attention method'
for path in (REPO, ATTENTION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attention_ecg.cepstral import CepstralConfig, cepstral_features  # noqa: E402
from attention_ecg import ecgdata as v1data_module  # noqa: E402
from attention_ecg.ecgdata import (file_hash, array_hash, read_json, atomic_json, atomic_npz,  # noqa: E402
                                   writer_lock, validate_fold_arrays)

PROTOCOL = ROOT/'protocol.json'
CACHE = ROOT/'features/all_windows'
RUN = ROOT/'results/all_windows'
REPORTS = ROOT/'reports'
V1_PROTOCOL = ATTENTION/'ecgdata_protocol.json'
V1_CACHE = ATTENTION/'features/ecgdata_lfcc_v1'
V1_RUN = ATTENTION/'results/ecgdata_lfcc_swin_v1'


def log(message):
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {message}', flush=True)


def protocol():
    return read_json(PROTOCOL)


def sources():
    paths = [PROTOCOL, Path(__file__), ATTENTION/'attention_ecg/cepstral.py', ATTENTION/'attention_ecg/model.py',
             ATTENTION/'attention_ecg/ecgdata.py', ATTENTION/'attention_ecg/ecgdata_training.py',
             ATTENTION/'attention_ecg/ecgdata_report.py', REPO/'ecgvmd/data.py', REPO/'ecgvmd/segment.py',
             REPO/'ecgvmd/ecgdata_subjects.csv', REPO/'ecgvmd/ecgdata_subjects.json']
    return {str(p.relative_to(REPO)): file_hash(p) for p in paths}


def grid_windows(data, seg_len):
    """Every non-overlapping window, row-major, z-scored exactly as the v1 baseline."""
    from ecgvmd.segment import fixed_windows, standardise
    rows = np.arange(data.shape[0])
    windows, row_labels, row_ids = fixed_windows(data, rows, rows, seg_len=seg_len, n_per_record=0)
    per_row = data.shape[1]//seg_len
    starts = np.tile(np.arange(per_row)*seg_len, data.shape[0])
    if not np.array_equal(row_ids, np.repeat(rows, per_row)):
        raise ValueError('Unexpected grid order')
    return standardise(windows), row_ids.astype(np.int64), starts.astype(np.int64)


def patient_folds(patients, folds):
    """Patient -> fold from the v1 per-window assignment; rejects any crossing."""
    mapping = {}
    for patient, fold in zip(patients, folds):
        if mapping.setdefault(str(patient), int(fold)) != int(fold):
            raise ValueError(f'Patient {patient} crosses v1 folds')
    return mapping


def subset_positions(row_ids, starts, seg_len, per_row):
    """Grid index of each v1 window (row, start)."""
    if np.any(starts % seg_len) or np.any(starts//seg_len >= per_row):
        raise ValueError('v1 window is off the non-overlapping grid')
    return row_ids*per_row + starts//seg_len


def load_v1():
    manifest, data = v1data_module.verify_cache(V1_CACHE)
    return manifest, data


def build(p):
    from ecgvmd.data import load_ecgdata
    mat = REPO/p['data']['mat']
    if file_hash(mat) != p['data']['mat_sha256']:
        raise ValueError('ECGData.mat differs from the protocol')
    dataset = load_ecgdata(str(mat))
    seg_len = p['data']['seg_len']
    windows, row_ids, starts = grid_windows(dataset.data, seg_len)
    per_row = dataset.n_samples//seg_len
    if per_row != p['data']['windows_per_row'] or len(windows) != p['data']['n_windows']:
        raise ValueError('Window count differs from protocol')
    v1_manifest, v1 = load_v1()
    if v1_manifest['mat_sha256'] != p['data']['mat_sha256']:
        raise ValueError('v1 cache used another ECGData.mat')
    folds_by_patient = patient_folds(v1['patients'], v1['fold'])
    patients = dataset.patient_ids[row_ids].astype(str)
    labels = dataset.labels[row_ids].astype(str)
    fold = np.array([folds_by_patient[x] for x in patients], dtype=np.int64)
    y = np.array([p['classes'].index(x) for x in labels], dtype=np.int64)
    validate_fold_arrays(y, patients, fold)
    positions = subset_positions(v1['row_ids'], v1['window_start'], seg_len, per_row)
    if len(np.unique(positions)) != len(positions):
        raise ValueError('Duplicate v1 window positions')
    for key, local in [('y', y), ('patients', patients), ('row_ids', row_ids), ('window_start', starts), ('fold', fold)]:
        if not np.array_equal(local[positions], v1[key]):
            raise ValueError(f'v1 subset {key} does not align')
    # Exact sample-level identity with the frozen baseline windows (hash-checked there).
    v1_windows, _ = v1data_module.matched_windows(mat, v1_manifest['protocol'])
    if not np.array_equal(windows[positions], v1_windows):
        raise ValueError('Grid windows differ from v1 windows')
    v1_index = np.full(len(windows), -1, dtype=np.int64)
    v1_index[positions] = np.arange(len(positions))
    arrays = dict(y=y, labels=labels, patients=patients, sources=dataset.source_ids[row_ids].astype(str),
                  row_ids=row_ids, fold=fold, window_start=starts, window_id=np.arange(len(y), dtype=np.int64),
                  v1_index=v1_index)
    return windows, arrays, v1


def prepare():
    p = protocol()
    start = time.perf_counter()
    with writer_lock(CACHE):
        if (CACHE/'completed.json').exists():
            verify_cache()
            log('Cache already complete and verified; nothing extracted')
            return dict(status='reused')
        windows, arrays, v1 = build(p)
        log(f'Built {len(windows)} grid windows; v1 subset identity verified ({len(v1["y"])} windows)')
        manifest = dict(protocol_sha256=file_hash(PROTOCOL), sources=sources(), windows_sha256=array_hash(windows),
                        fold_sha256=array_hash(arrays['fold']), v1_cache_completed_sha256=file_hash(V1_CACHE/'completed.json'),
                        versions=dict(python=platform.python_version(), numpy=np.__version__))
        if (CACHE/'manifest.json').exists():
            if read_json(CACHE/'manifest.json') != manifest:
                raise ValueError('Interrupted cache inputs changed: use a new directory')
        else:
            atomic_json(CACHE/'manifest.json', manifest)
        config = CepstralConfig(**p['cepstral'])
        shards = CACHE/'shards'
        shards.mkdir(exist_ok=True)
        blocks, inventory, reused = [], {}, 0
        for row in np.unique(arrays['row_ids']):
            idx = np.flatnonzero(arrays['row_ids'] == row)
            artifact, marker = shards/f'row_{row:03d}.npz', shards/f'row_{row:03d}.json'
            if marker.exists():
                saved = read_json(marker)
                if saved['sha256'] != file_hash(artifact) or saved['windows_sha256'] != array_hash(windows[idx]):
                    raise ValueError(f'Corrupt or mismatched shard {row}')
                with np.load(artifact, allow_pickle=False) as z:
                    block = z['X']
                reused += 1
            else:
                began = time.perf_counter()
                extracted = [cepstral_features(windows[i, :, None], config) for i in idx]
                block = np.stack([e[0] for e in extracted])
                atomic_npz(artifact, X=block, indices=idx)
                atomic_json(marker, dict(sha256=file_hash(artifact), windows_sha256=array_hash(windows[idx]),
                                         seconds=time.perf_counter()-began, metadata=extracted[0][1]))
            if block.shape != (len(idx), 13, 1, 16) or not np.isfinite(block).all():
                raise ValueError('Unexpected or nonfinite shard')
            blocks.append(block)
            inventory[f'shards/{artifact.name}'] = file_hash(artifact)
            inventory[f'shards/{marker.name}'] = file_hash(marker)
        X = np.concatenate(blocks)
        subset = arrays['v1_index'] >= 0
        order = np.argsort(arrays['v1_index'][subset])
        if not np.array_equal(X[subset][order], v1['X']):
            raise ValueError('LFCC features of v1 windows differ from the v1 cache')
        log('LFCC of the v1 subset is bit-identical to the v1 cache')
        atomic_npz(CACHE/'features.npz', X=X, **arrays)
        inventory['features.npz'] = file_hash(CACHE/'features.npz')
        counts = {c: dict(windows=int((arrays['y'] == i).sum()),
                          patients=len(np.unique(arrays['patients'][arrays['y'] == i])))
                  for i, c in enumerate(p['classes'])}
        summary = dict(status='complete', manifest_sha256=file_hash(CACHE/'manifest.json'), files=inventory,
                       shape=list(X.shape), patients=len(np.unique(arrays['patients'])), rows=int(len(np.unique(arrays['row_ids']))),
                       class_counts=counts, v1_subset_windows=int(subset.sum()), v1_features_identical=True,
                       v1_windows_identical=True, reused_shards=reused, seconds=time.perf_counter()-start,
                       training_started=False)
        atomic_json(CACHE/'completed.json', summary)
    verify_cache()
    return {k: v for k, v in summary.items() if k != 'files'}


def verify_cache():
    completed = read_json(CACHE/'completed.json')
    if completed['status'] != 'complete' or completed['manifest_sha256'] != file_hash(CACHE/'manifest.json'):
        raise ValueError('Incomplete or altered cache')
    for name, digest in completed['files'].items():
        if file_hash(CACHE/name) != digest:
            raise ValueError(f'Cache artifact changed: {name}')
    if read_json(CACHE/'manifest.json')['sources'] != sources():
        raise ValueError('Source changed since extraction: use a new study directory')
    with np.load(CACHE/'features.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if data['X'].shape != tuple(completed['shape']) or not np.isfinite(data['X']).all():
        raise ValueError('Invalid feature matrix')
    validate_fold_arrays(data['y'], data['patients'], data['fold'])
    return data


def run_names(p):
    for fold in range(5):
        yield 'logistic', None, fold, f'logistic_fold{fold}'
        for seed in p['training']['seeds']:
            yield 'swin', seed, fold, f'swin_seed{seed}_fold{fold}'


def train(execute=False, device='cuda'):
    from attention_ecg.ecgdata_training import require_training, fit_control, fit_swin, verify_trial, complete_trial
    require_training(execute)
    import sklearn
    import torch
    p = protocol()
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable; no silent CPU fallback')
    data = verify_cache()
    torch.set_num_threads(p['training']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA training')
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    manifest = dict(protocol_sha256=file_hash(PROTOCOL), cache_completed_sha256=file_hash(CACHE/'completed.json'),
                    sources=sources(), device=device,
                    device_name=torch.cuda.get_device_name() if device == 'cuda' else platform.processor(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                                  torch=torch.__version__, cuda=torch.version.cuda))
    with writer_lock(RUN):
        if (RUN/'manifest.json').exists():
            if read_json(RUN/'manifest.json') != manifest:
                raise ValueError('Training manifest differs; use a new run directory')
        else:
            atomic_json(RUN/'manifest.json', manifest)
        if (RUN/'completed.json').exists():
            if not verify_trial(RUN) or not all(verify_trial(RUN/name) for *_, name in run_names(p)):
                raise ValueError('Invalid completed run')
            log('Run already complete and verified; nothing trained')
            return dict(status='reused')
        atomic_json(RUN/'status.json', dict(status='running', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                            pid=os.getpid()))
        try:
            began = time.perf_counter()
            for arm, seed, fold, name in run_names(p):
                folder = RUN/name
                if verify_trial(folder):
                    log(f'{name}: already complete, verified and reused')
                    continue
                t = time.perf_counter()
                if arm == 'logistic':
                    fit_control(folder, data, fold, p)
                else:
                    fit_swin(folder, data, fold, seed, p, device)
                log(f'{name}: complete in {time.perf_counter()-t:.1f}s')
                atomic_json(RUN/'status.json', dict(status='running', last_completed=name, pid=os.getpid()))
            names = ['manifest.json'] + [f'{name}/completed.json' for *_, name in run_names(p)]
            complete_trial(RUN, names)
            atomic_json(RUN/'status.json', dict(status='complete', trials=20, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                session_seconds=time.perf_counter()-began))
        except BaseException as exc:
            atomic_json(RUN/f'failure_{time.time_ns()}.json', dict(type=type(exc).__name__, message=str(exc),
                                                                  traceback=traceback.format_exc()))
            atomic_json(RUN/'status.json', dict(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'))
            raise
    return dict(status='complete', trials=20)


# ---------------------------------------------------------------- reporting
def summarize(y, patients, pred):
    from attention_ecg.ecgdata_report import patient_confusions, scores
    windows, votes = patient_confusions(y, pred, patients)
    return dict(window=scores(windows.sum(0)), patient_vote=scores(votes.sum(0)))


def paired_bootstrap(y, patients, arms, pairs, repetitions, seed):
    """Same patient multiplicities for every arm; seeds averaged inside each draw."""
    from attention_ecg.ecgdata_report import patient_confusions, metric_vector
    grouped = {arm: [patient_confusions(y, pred, patients) for pred in preds] for arm, preds in arms.items()}
    n = len(np.unique(patients))
    rng = np.random.default_rng(seed)
    reference = next(iter(grouped.values()))[0][0]
    draws, rejected = {arm: [] for arm in arms}, 0
    for _ in range(repetitions):
        mult = np.bincount(rng.integers(n, size=n), minlength=n)
        if np.any((reference*mult[:, None, None]).sum((0, 2)) == 0):
            rejected += 1
            continue
        for arm, items in grouped.items():
            draws[arm].append(np.mean([metric_vector((w*mult[:, None, None]).sum(0), (v*mult[:, None, None]).sum(0))
                                       for w, v in items], axis=0))
    keys = ['window_macro_f1', 'window_accuracy', 'patient_vote_macro_f1', 'patient_vote_accuracy']
    draws = {arm: np.array(v) for arm, v in draws.items()}
    intervals = {arm: {k: np.quantile(v[:, i], [.025, .975]).tolist() for i, k in enumerate(keys)} for arm, v in draws.items()}
    differences = {}
    for a, b in pairs:
        diff = draws[a]-draws[b]
        differences[f'{a}_minus_{b}'] = {k: dict(interval_95=np.quantile(diff[:, i], [.025, .975]).tolist(),
                                                 fraction_positive=float((diff[:, i] > 0).mean()))
                                         for i, k in enumerate(keys)}
    return dict(repetitions=repetitions, seed=seed, valid_draws=repetitions-rejected,
                rejected_single_class_draws=rejected, intervals_95=intervals, paired_differences=differences)


def mean_metrics(per_seed):
    return {scope: {m: float(np.mean([r[scope][m] for r in per_seed])) for m in ('accuracy', 'macro_f1', 'balanced_accuracy')}
            | {'sensitivity': np.mean([r[scope]['sensitivity'] for r in per_seed], axis=0).tolist()}
            for scope in ('window', 'patient_vote')}


def report():
    from attention_ecg.ecgdata_report import collect_predictions
    from attention_ecg.ecgdata_training import verify_trial
    p = protocol()
    data = verify_cache()
    if not verify_trial(RUN):
        raise ValueError('All 20 fits must finish before reporting')
    if read_json(RUN/'manifest.json')['cache_completed_sha256'] != file_hash(CACHE/'completed.json'):
        raise ValueError('Run used another cache')
    new = collect_predictions(RUN, data, p)
    v1_manifest, v1 = load_v1()
    old = collect_predictions(V1_RUN, v1, v1_manifest['protocol'])
    subset = data['v1_index'] >= 0
    pos = np.flatnonzero(subset)[np.argsort(data['v1_index'][subset])]  # grid index in v1 window order
    if not (np.array_equal(data['y'][pos], v1['y']) and np.array_equal(data['patients'][pos], v1['patients'])):
        raise ValueError('Subset alignment failed')
    sets = dict(
        v1_windows=dict(y=v1['y'], patients=v1['patients'],
                        arms={'all_swin': [x[pos] for x in new['swin']], 'all_logistic': [x[pos] for x in new['logistic']],
                              'v1_swin': old['swin'], 'v1_logistic': old['logistic']},
                        pairs=[('all_swin', 'v1_swin'), ('all_logistic', 'v1_logistic'), ('all_swin', 'all_logistic'),
                               ('v1_swin', 'v1_logistic')]),
        all_windows=dict(y=data['y'], patients=data['patients'],
                         arms={'all_swin': new['swin'], 'all_logistic': new['logistic']},
                         pairs=[('all_swin', 'all_logistic')]))
    e = p['evaluation']
    results = {}
    for name, s in sets.items():
        arms = {arm: dict(per_seed=[summarize(s['y'], s['patients'], x) for x in preds]) for arm, preds in s['arms'].items()}
        for arm in arms.values():
            arm['mean'] = mean_metrics(arm['per_seed'])
        results[name] = dict(n_windows=len(s['y']), arms=arms,
                             bootstrap=paired_bootstrap(s['y'], s['patients'], s['arms'], s['pairs'],
                                                        e['bootstrap_resamples'], e['bootstrap_seed']))
    fits = []
    for arm, seed, fold, name in run_names(p):
        fit = read_json(RUN/name/'fit.json')
        row = dict(trial=name, arm=arm, seed=seed, fold=fold, train_windows=len(fit['train_indices']),
                   test_windows=len(fit['test_indices']), seconds=fit['seconds'])
        if arm == 'swin':
            history = read_json(RUN/name/'history.json')
            row.update(final_loss=history[-1]['loss'], epochs=len(history))
        else:
            row.update(warnings=len(fit['warnings']), iterations=max(fit['iterations']))
        fits.append(row)
    analysis = dict(protocol_sha256=file_hash(PROTOCOL), run_manifest_sha256=file_hash(RUN/'manifest.json'),
                    cache_completed_sha256=file_hash(CACHE/'completed.json'), v1_run_completed_sha256=file_hash(V1_RUN/'completed.json'),
                    classes=p['classes'], seeds=p['training']['seeds'], results=results, fits=fits,
                    uncertainty_scope=e['interval_scope'], report_source_sha256=file_hash(Path(__file__)))
    REPORTS.mkdir(exist_ok=True)
    atomic_json(REPORTS/'analysis.json', analysis)
    atomic_npz(RUN/'predictions_export.npz', y=data['y'], patients=data['patients'], row_ids=data['row_ids'],
               window_start=data['window_start'], fold=data['fold'], v1_index=data['v1_index'],
               **{f'swin_seed{s}': x for s, x in zip(p['training']['seeds'], new['swin'])}, logistic=new['logistic'][0])
    with (REPORTS/'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['evaluation_set', 'arm', 'seed', 'scope', 'accuracy', 'macro_f1', 'balanced_accuracy',
                         'recall_ARR', 'recall_CHF', 'recall_NSR'])
        for set_name, r in results.items():
            for arm, a in r['arms'].items():
                seeds = p['training']['seeds'] if 'swin' in arm else ['deterministic']
                for seed, item in zip(seeds, a['per_seed']):
                    for scope in ('window', 'patient_vote'):
                        m = item[scope]
                        writer.writerow([set_name, arm, seed, scope, f"{m['accuracy']:.6f}", f"{m['macro_f1']:.6f}",
                                         f"{m['balanced_accuracy']:.6f}", *[f'{x:.6f}' for x in m['sensitivity']]])
    with (REPORTS/'fits.csv').open('w', newline='') as stream:
        keys = sorted({k for f in fits for k in f})
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(fits)
    write_markdown(analysis)
    return dict(status='reported', report=str(REPORTS/'report.md'))


def write_markdown(a):
    pct = lambda x: f'{100*x:.2f}%'
    lines = ['# ECG-07 all-window LFCC/Swin — results', '',
             'Generated by `study.py report` from saved out-of-fold predictions. Swin rows are means over seeds 0/1/2; '
             'logistic is deterministic. Intervals: 2,000 paired patient-bootstrap draws (seed 20260925), pointwise, '
             'conditional on saved predictions.', '']
    for set_name, title in [('v1_windows', 'Matched evaluation on the 1,620 v1 windows (primary)'),
                            ('all_windows', 'All 21,222 held-out windows')]:
        r = a['results'][set_name]
        lines += [f'## {title}', '', '| Arm | Window accuracy | Window macro-F1 | Patient-vote accuracy | Patient-vote macro-F1 | Recall ARR / CHF / NSR (window) |',
                  '|---|---:|---:|---:|---:|---|']
        for arm, item in r['arms'].items():
            w, v = item['mean']['window'], item['mean']['patient_vote']
            lines.append(f"| {arm} | {pct(w['accuracy'])} | {w['macro_f1']:.4f} | {pct(v['accuracy'])} | {v['macro_f1']:.4f} | "
                         + ' / '.join(pct(x) for x in w['sensitivity']) + ' |')
        lines += ['', '| Paired difference | Metric | Observed | 95% interval |', '|---|---|---:|---:|']
        for pair, metrics in r['bootstrap']['paired_differences'].items():
            a_name, b_name = pair.split('_minus_')
            for key, d in metrics.items():
                scope, metric = ('window', key[7:]) if key.startswith('window_') else ('patient_vote', key[14:])
                observed = r['arms'][a_name]['mean'][scope][metric]-r['arms'][b_name]['mean'][scope][metric]
                lo, hi = d['interval_95']
                fmt = (lambda x: f'{100*x:+.2f} pp') if metric == 'accuracy' else (lambda x: f'{x:+.4f}')
                lines.append(f'| {a_name} − {b_name} | {key} | {fmt(observed)} | [{fmt(lo)}, {fmt(hi)}] |')
        lines += ['', f"Valid bootstrap draws: {r['bootstrap']['valid_draws']}/{r['bootstrap']['repetitions']}.", '']
    total = sum(f['seconds'] for f in a['fits'])
    lines += ['## Fits', '', f"20 fits; summed fit time {total:.1f} s. Per-fit rows: [fits.csv](fits.csv). "
              'All metrics per seed: [metrics.csv](metrics.csv). Full analysis: [analysis.json](analysis.json).', '']
    (REPORTS/'report.md').write_text('\n'.join(lines))
