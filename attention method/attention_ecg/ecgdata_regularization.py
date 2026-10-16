"""Patient-nested LFCC/Swin study. Frozen v1 inputs are read-only."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
import torch

from .ecgdata import (ROOT, REPO, DEFAULT_CACHE, verify_cache, read_json, file_hash,
                      atomic_json, atomic_npz, fit_normalizer, normalize, writer_lock)
from .ecgdata_training import (DEFAULT_RUN, partition, batch_indices,
                               verify_trial, complete_trial, load_checkpoint)
from .ecgdata_report import (scores, confusion, patient_confusions,
                            bootstrap_comparison, collect_predictions)
from .regularized_swin import make_model

SPEC = ROOT / 'ecgdata_regularization_protocol_v2.json'
RUN = ROOT / 'results/ecgdata_lfcc_swin_regularization_v2'
REPORT = ROOT / 'reports/ecgdata_lfcc_swin_regularization_v2'


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def setup(device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty CSV')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inner_splits(data, outer_fold, n_splits, seed):
    train, test = partition(data, outer_fold)
    patients = np.unique(data['patients'][train])
    labels = []
    for patient in patients:
        values = np.unique(data['y'][train[data['patients'][train] == patient]])
        if len(values) != 1:
            raise ValueError('Inconsistent patient labels')
        labels.append(int(values[0]))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + outer_fold)
    result = []
    seen = np.zeros(len(data['y']), dtype=int)
    for fit_patients, val_patients in splitter.split(patients, labels):
        fit = train[np.isin(data['patients'][train], patients[fit_patients])]
        val = train[np.isin(data['patients'][train], patients[val_patients])]
        groups = [set(data['patients'][indices]) for indices in (fit, val, test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError('Patient leakage')
        if set(data['y'][fit]) != {0, 1, 2} or set(data['y'][val]) != {0, 1, 2}:
            raise ValueError('Missing inner class')
        seen[val] += 1
        result.append((fit, val))
    if not np.all(seen[train] == 1) or np.any(seen[test]):
        raise ValueError('Inner coverage invalid')
    return result


def predict(model, X, indices, device, batch_size):
    model.eval()
    with torch.inference_mode():
        values = [model(torch.from_numpy(X[indices[i:i + batch_size]]).to(device))
                  .softmax(-1).cpu().numpy() for i in range(0, len(indices), batch_size)]
    output = np.concatenate(values)
    if not np.isfinite(output).all() or not np.allclose(output.sum(1), 1):
        raise ValueError('Invalid probabilities')
    return output


def save_state(folder, state):
    """Immutable generations + atomic pointer survive a pre-pointer interruption."""
    rolling = folder / 'rolling'
    rolling.mkdir(exist_ok=True)
    path = rolling / f'epoch_{state["epoch"]:03d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)
    atomic_json(folder / 'checkpoint.json', dict(epoch=state['epoch'],
                path=str(path.relative_to(folder)), sha256=file_hash(path)))
    # Only our disposable checkpoint generations are removed. Candidate weights
    # are separately retained; keep the current and previous rolling generations.
    for previous in rolling.glob('epoch_*.pt'):
        if int(previous.stem.split('_')[1]) < state['epoch'] - 1:
            previous.unlink()


def load_state(folder):
    pointer = folder / 'checkpoint.json'
    if not pointer.exists():
        return None
    marker = read_json(pointer)
    path = folder / marker['path']
    if folder.resolve() not in path.resolve().parents or file_hash(path) != marker['sha256']:
        raise ValueError('Corrupt checkpoint')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['epoch'] != marker['epoch']:
        raise ValueError('Checkpoint epoch mismatch')
    return state


def save_candidate(folder, epoch, model, probabilities, data, indices):
    stem = f'epoch_{epoch:03d}'
    path = folder / (stem + '.pt')
    temporary = path.with_suffix('.tmp')
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, temporary)
    os.replace(temporary, path)
    atomic_npz(folder / (stem + '.npz'), indices=indices, y=data['y'][indices],
               patients=data['patients'][indices], probabilities=probabilities,
               pred=probabilities.argmax(1))
    atomic_json(folder / (stem + '.json'), dict(epoch=epoch, files={
        name: file_hash(folder / name) for name in (stem + '.pt', stem + '.npz')}))


def fit(folder, data, train, evaluation, seed, dropout, epochs, retained,
        base, device, signature, kind, stop_after=None):
    """Inner validation each epoch; outer evaluation only at the selected end."""
    folder = Path(folder)
    identity = dict(signature=signature, kind=kind, seed=seed, dropout=dropout,
                    epochs=epochs, retained=retained, train=train.tolist(),
                    evaluation=evaluation.tolist(), model=base['model'], training=base['training'])
    if set(data['patients'][train]) & set(data['patients'][evaluation]):
        raise ValueError('Training/evaluation patient overlap')
    if (folder / 'identity.json').exists() and read_json(folder / 'identity.json') != identity:
        raise ValueError('Trial identity changed')
    if verify_trial(folder):
        return read_json(folder / 'fit.json')
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / 'identity.json', identity)
    settings = base['training']
    mean, scale = fit_normalizer(data['X'], train)
    X = normalize(data['X'], mean, scale)
    atomic_npz(folder / 'preprocessing.npz', mean=mean, scale=scale,
               train_indices=train, evaluation_indices=evaluation)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = make_model(base['model'], dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                      T_max=settings['epochs'], eta_min=settings['min_lr'])
    counts = np.bincount(data['y'][train], minlength=3)
    weights = len(train) / (3 * counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    history, first, prior_seconds = [], 0, 0.0
    state = load_state(folder)
    if state is not None:
        if state['identity_sha256'] != digest(identity):
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, first, prior_seconds = state['history'], state['epoch'], state['seconds']
        if first > epochs or len(history) != first:
            raise ValueError('Invalid checkpoint progress')
        torch.set_rng_state(state['rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    started = time.perf_counter()
    for epoch in range(first, epochs):
        model.train()
        losses, masses = [], []
        lr = optimizer.param_groups[0]['lr']
        for batch in batch_indices(len(train), settings['batch_size'], seed, epoch,
                                   settings['shuffle_seed_offset']):
            indices = train[batch]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(torch.from_numpy(X[indices]).to(device)),
                             torch.from_numpy(data['y'][indices]).to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'],
                                          error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            masses.append(float(weights[data['y'][indices]].sum()))
        scheduler.step()
        row = dict(epoch=epoch + 1, loss=float(np.average(losses, weights=masses)), lr=lr)
        if kind == 'inner' or epoch + 1 == epochs:
            probabilities = predict(model, X, evaluation, device, settings['batch_size'])
            metric = scores(confusion(data['y'][evaluation], probabilities.argmax(1)))
            row.update(evaluation_accuracy=metric['accuracy'], evaluation_macro_f1=metric['macro_f1'])
            if epoch + 1 in retained:
                save_candidate(folder, epoch + 1, model, probabilities, data, evaluation)
        history.append(row)
        state = dict(epoch=epoch + 1, identity_sha256=digest(identity),
                     model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     history=history, seconds=prior_seconds + time.perf_counter() - started,
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [])
        save_state(folder, state)
        atomic_json(folder / 'history.json', history)
        if stop_after is not None and epoch + 1 >= stop_after:
            return dict(status='interrupted_for_test', epoch=epoch + 1)
    # Check every retained checkpoint and saved prediction by a fresh model load.
    checks = []
    for epoch in retained:
        stem = f'epoch_{epoch:03d}'
        marker = read_json(folder / (stem + '.json'))
        for name, expected in marker['files'].items():
            if file_hash(folder / name) != expected:
                raise ValueError('Changed candidate checkpoint')
        reloaded = make_model(base['model'], dropout).to(device)
        reloaded.load_state_dict(torch.load(folder / (stem + '.pt'), map_location='cpu', weights_only=True))
        actual = predict(reloaded, X, evaluation, device, settings['batch_size'])
        with np.load(folder / (stem + '.npz'), allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['indices'], evaluation)
            np.testing.assert_array_equal(saved['y'], data['y'][evaluation])
            np.testing.assert_allclose(actual, saved['probabilities'], rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(actual.argmax(1), saved['pred'])
            checks.append(dict(epoch=epoch, maximum_probability_error=float(abs(actual - saved['probabilities']).max())))
    training_probabilities = predict(model, X, train, device, settings['batch_size'])
    record = dict(status='complete', seed=seed, dropout=dropout, epochs=epochs,
                  seconds=prior_seconds + time.perf_counter() - started, kind=kind,
                  train_patients=len(np.unique(data['patients'][train])), train_windows=len(train),
                  evaluation_patients=len(np.unique(data['patients'][evaluation])), evaluation_windows=len(evaluation),
                  class_weights=weights.tolist(), parameter_count=sum(p.numel() for p in model.parameters()),
                  training=scores(confusion(data['y'][train], training_probabilities.argmax(1))),
                  reload_checks=checks, peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                  warnings=[])
    atomic_json(folder / 'fit.json', record)
    atomic_json(folder / 'history.json', history)
    names = ['identity.json', 'preprocessing.npz', 'checkpoint.json', 'history.json', 'fit.json']
    names += [str(p.relative_to(folder)) for p in sorted(folder.glob('epoch_*.*'))]
    names += [read_json(folder / 'checkpoint.json')['path']]
    complete_trial(folder, names)
    return record


def choose(rows):
    return min(rows, key=lambda row: (-row['mean_macro_f1'], row['epoch'], row['candidate']))


def reuse_v1(folder, data, outer, seed, base, signature, device, source=DEFAULT_RUN):
    """An R0 choice of epoch 40 is the completed v1 fit; never retrain it."""
    started = time.perf_counter()
    folder = Path(folder)
    original = source / f'swin_seed{seed}_fold{outer}'
    if not verify_trial(original):
        raise ValueError('Missing reusable v1 fit')
    identity = dict(signature=signature, kind='outer', seed=seed, dropout=0.0,
                    reuse_source=str(original), source_marker_sha256=file_hash(original / 'completed.json'))
    if (folder / 'identity.json').exists() and read_json(folder / 'identity.json') != identity:
        raise ValueError('Reuse identity changed')
    if verify_trial(folder):
        return read_json(folder / 'fit.json')
    folder.mkdir(parents=True, exist_ok=True)
    train, evaluation = partition(data, outer)
    mean, scale = fit_normalizer(data['X'], train)
    X = normalize(data['X'], mean, scale)
    saved_fit = read_json(original / 'fit.json')
    if saved_fit['train_indices'] != train.tolist() or saved_fit['test_indices'] != evaluation.tolist():
        raise ValueError('Reusable v1 partition differs')
    with np.load(original / 'preprocessing.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['mean'], mean)
        np.testing.assert_array_equal(z['scale'], scale)
    state = load_checkpoint(original)
    epoch = base['training']['epochs']
    if state['epoch'] != epoch or state['seed'] != seed or state['fold'] != outer:
        raise ValueError('Reusable v1 checkpoint differs')
    model = make_model(base['model'], 0).to(device)
    model.load_state_dict(state['model'])
    actual = predict(model, X, evaluation, device, base['training']['batch_size'])
    with np.load(original / 'predictions.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'], evaluation)
        np.testing.assert_allclose(actual, z['probabilities'], rtol=2e-4, atol=2e-5)
        np.testing.assert_array_equal(actual.argmax(1), z['pred'])
        probability_error = float(abs(actual - z['probabilities']).max())
        save_candidate(folder, epoch, model, z['probabilities'], data, evaluation)
    atomic_json(folder / 'identity.json', identity)
    atomic_npz(folder / 'preprocessing.npz', mean=mean, scale=scale,
               train_indices=train, evaluation_indices=evaluation)
    history = [dict(epoch=r['epoch'], loss=r['loss'], lr=r['lr']) for r in state['history']]
    final_metric = scores(confusion(data['y'][evaluation], actual.argmax(1)))
    history[-1].update(evaluation_accuracy=final_metric['accuracy'], evaluation_macro_f1=final_metric['macro_f1'])
    atomic_json(folder / 'history.json', history)
    train_probabilities = predict(model, X, train, device, base['training']['batch_size'])
    record = dict(status='complete', kind='outer', seed=seed, dropout=0.0, epochs=epoch,
                  reused_v1=True, original_training_seconds=saved_fit['seconds'],
                  seconds=time.perf_counter() - started, train_patients=len(np.unique(data['patients'][train])),
                  train_windows=len(train), evaluation_patients=len(np.unique(data['patients'][evaluation])),
                  evaluation_windows=len(evaluation), class_weights=saved_fit['class_weights'],
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  training=scores(confusion(data['y'][train], train_probabilities.argmax(1))),
                  reload_checks=[dict(epoch=epoch, maximum_probability_error=probability_error)],
                  peak_cuda_bytes=0, warnings=[])
    atomic_json(folder / 'fit.json', record)
    complete_trial(folder, [p.name for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'completed.json'])
    return record


def select_candidates(run, data, outer, spec):
    train, _ = partition(data, outer)
    rows = []
    for candidate in spec['candidates']:
        for epoch in spec['epoch_candidates']:
            seed_scores = []
            for seed in spec['seeds']:
                predictions = np.full(len(data['y']), -1, dtype=int)
                coverage = np.zeros(len(data['y']), dtype=int)
                for inner in range(spec['inner_folds']):
                    folder = run / 'inner' / f'outer{outer}_{candidate["id"]}_inner{inner}_seed{seed}'
                    if not verify_trial(folder):
                        raise ValueError('Inner fit incomplete')
                    with np.load(folder / f'epoch_{epoch:03d}.npz', allow_pickle=False) as z:
                        idx = z['indices']
                        if not np.isin(idx, train).all():
                            raise ValueError('Outer patient entered selection')
                        np.testing.assert_array_equal(z['y'], data['y'][idx])
                        predictions[idx] = z['pred']
                        coverage[idx] += 1
                if not np.all(coverage[train] == 1) or coverage.sum() != len(train):
                    raise ValueError('Inner prediction coverage differs')
                seed_scores.append(scores(confusion(data['y'][train], predictions[train]))['macro_f1'])
            rows.append(dict(candidate=candidate['id'], epoch=epoch,
                             seed_macro_f1=seed_scores, mean_macro_f1=float(np.mean(seed_scores))))
    return dict(outer_fold=outer, rows=rows, selected=choose(rows), per_candidate={
        candidate['id']: choose([r for r in rows if r['candidate'] == candidate['id']])
        for candidate in spec['candidates']})


def protected_files():
    paths = set()
    for base in (DEFAULT_CACHE, DEFAULT_RUN, ROOT / 'reports/ecgdata_lfcc_swin_v1'):
        paths.update(p for p in base.rglob('*') if p.is_file() and p.name != '.writer.lock')
    paths.update(ROOT / p for p in ['ecgdata_protocol.json', 'attention_ecg/model.py',
        'attention_ecg/cepstral.py', 'attention_ecg/ecgdata.py', 'attention_ecg/ecgdata_training.py',
        'attention_ecg/ecgdata_report.py'])
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)}


def check_protected(expected):
    for name, sha in expected.items():
        if file_hash(REPO / name) != sha:
            raise ValueError(f'Protected file changed: {name}')


def prepare(device):
    manifest, data = verify_cache()
    if not verify_trial(DEFAULT_RUN):
        raise ValueError('v1 run incomplete')
    # Independent saved inventory check, not just mutable completion markers.
    with (ROOT / 'reports/ecgdata_lfcc_swin_v1/artifact_inventory.csv').open(newline='') as stream:
        inventory = list(csv.DictReader(stream))
    for item in inventory:
        path = DEFAULT_RUN / item['path']
        if path.stat().st_size != int(item['bytes']) or file_hash(path) != item['sha256']:
            raise ValueError(f'v1 inventory mismatch: {item["path"]}')
    collect_predictions(DEFAULT_RUN, data, manifest['protocol'])
    spec = read_json(SPEC)
    base = manifest['protocol']
    if base['training']['epochs'] != spec['schedule_horizon']:
        raise ValueError('Cosine horizon changed')
    source_paths = [Path(__file__), ROOT / 'attention_ecg/regularized_swin.py', SPEC,
                    ROOT / 'ecgdata_regularization_protocol_v2.md',
                    ROOT / 'scripts/run_ecgdata_regularization.py',
                    ROOT / 'tests/test_ecgdata_regularization.py']
    source_paths += [ROOT / 'attention_ecg' / name for name in
                     ['model.py', 'ecgdata.py', 'ecgdata_training.py', 'ecgdata_report.py']]
    sources = {str(p.relative_to(REPO)): file_hash(p) for p in source_paths}
    identity = dict(spec=spec, base_protocol=base, sources=sources,
                    cache_marker_sha256=file_hash(DEFAULT_CACHE / 'completed.json'),
                    v1_inventory_sha256=file_hash(ROOT / 'reports/ecgdata_lfcc_swin_v1/artifact_inventory.csv'),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  sklearn=sklearn.__version__, torch=torch.__version__, cuda=torch.version.cuda),
                    device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None)
    if (RUN / 'manifest.json').exists():
        saved = read_json(RUN / 'manifest.json')
        if saved['identity'] != identity:
            raise ValueError('Run input/source/environment changed; preserve and use a new protocol')
        check_protected(saved['protected_files'])
        for path in source_paths:
            if file_hash(RUN / 'sources' / path.name) != sources[str(path.relative_to(REPO))]:
                raise ValueError('Source snapshot changed')
    else:
        saved = dict(identity=identity, prepared_at=utc(), protected_files=protected_files(),
                     checked_v1_artifact_entries=len(inventory))
        (RUN / 'sources').mkdir(exist_ok=True)
        for path in source_paths:
            target = RUN / 'sources' / path.name
            target.write_bytes(path.read_bytes())
        atomic_json(RUN / 'manifest.json', saved)
        splits, csv_rows = {}, []
        for outer in range(5):
            pairs = inner_splits(data, outer, spec['inner_folds'], spec['inner_split_seed'])
            splits[str(outer)] = []
            for inner, (train, val) in enumerate(pairs):
                entry = dict(inner_fold=inner, fit_indices=train.tolist(), validation_indices=val.tolist())
                for role, indices in [('fit', train), ('validation', val)]:
                    patients = np.unique(data['patients'][indices])
                    entry[role + '_patient_counts'] = [sum(data['y'][np.flatnonzero(data['patients'] == p)[0]] == c
                                                           for p in patients) for c in range(3)]
                    entry[role + '_patient_counts'] = list(map(int, entry[role + '_patient_counts']))
                    for patient in patients:
                        csv_rows.append(dict(outer_fold=outer, inner_fold=inner, role=role, patient=patient,
                            label=base['classes'][data['y'][np.flatnonzero(data['patients'] == patient)[0]]]))
                splits[str(outer)].append(entry)
        atomic_json(RUN / 'splits.json', splits)
        write_csv(REPORT / 'splits.csv', csv_rows)
    return saved, data


def run(device='cuda', execute=False):
    if not execute:
        raise ValueError('Explicit --execute-training required')
    setup(device)
    REPORT.mkdir(parents=True, exist_ok=True)
    with writer_lock(RUN):
        manifest, data = prepare(device)
        if verify_trial(RUN):
            print(json.dumps(dict(status='reused', report=str(REPORT / 'report.md'))), flush=True)
            return
        identity = manifest['identity']
        spec, base = identity['spec'], identity['base_protocol']
        signature = digest(identity)
        started = time.perf_counter()
        try:
            selections = []
            for outer in range(5):
                pairs = inner_splits(data, outer, spec['inner_folds'], spec['inner_split_seed'])
                for candidate in spec['candidates']:
                    for inner, (train, val) in enumerate(pairs):
                        for seed in spec['seeds']:
                            name = f'outer{outer}_{candidate["id"]}_inner{inner}_seed{seed}'
                            atomic_json(RUN / 'status.json', dict(status='running', stage='inner', trial=name, time=utc()))
                            result = fit(RUN / 'inner' / name, data, train, val, seed, candidate['dropout'],
                                spec['schedule_horizon'], spec['epoch_candidates'], base, device, signature, 'inner')
                            print(json.dumps(dict(stage='inner', trial=name, seconds=result['seconds'], time=utc())), flush=True)
                selection = select_candidates(RUN, data, outer, spec)
                selections.append(selection)
                atomic_json(RUN / f'selection_outer{outer}.json', selection)
            atomic_json(RUN / 'selections.json', selections)
            # No outer evaluation until every selection is fixed.
            for outer, selection in enumerate(selections):
                train, test = partition(data, outer)
                for candidate in spec['candidates']:
                    epoch = selection['per_candidate'][candidate['id']]['epoch']
                    for seed in spec['seeds']:
                        name = f'outer{outer}_{candidate["id"]}_seed{seed}'
                        atomic_json(RUN / 'status.json', dict(status='running', stage='outer', trial=name, time=utc()))
                        if candidate['id'] == 'R0' and epoch == base['training']['epochs']:
                            result = reuse_v1(RUN / 'outer' / name, data, outer, seed, base, signature, device)
                        else:
                            result = fit(RUN / 'outer' / name, data, train, test, seed, candidate['dropout'], epoch,
                                [epoch], base, device, signature, 'outer')
                        print(json.dumps(dict(stage='outer', trial=name, epoch=epoch, seconds=result['seconds'], time=utc())), flush=True)
            atomic_json(RUN / 'status.json', dict(status='reporting', time=utc()))
            make_report(manifest, data)
            check_protected(manifest['protected_files'])
            check_protected(identity['sources'])
            atomic_json(RUN / 'verification.json', dict(status='passed', time=utc(),
                protected_files=len(manifest['protected_files']), current_source_files=len(identity['sources']),
                saved_candidate_and_final_checkpoints_reloaded=True,
                fit_count=len(list((RUN / 'inner').glob('*/completed.json'))) + len(list((RUN / 'outer').glob('*/completed.json'))),
                invocation_seconds=time.perf_counter() - started))
            immutable = [p for p in RUN.rglob('*') if p.is_file() and p.name != '.writer.lock'
                         and p.name not in ('completed.json', 'status.json') and 'failure_' not in p.name]
            immutable += list((RUN / 'inner').glob('*/completed.json')) + list((RUN / 'outer').glob('*/completed.json'))
            complete_trial(RUN, sorted({str(p.relative_to(RUN)) for p in immutable}))
            atomic_json(RUN / 'status.json', dict(status='complete', time=utc(), report=str(REPORT / 'report.md')))
            print(json.dumps(dict(status='complete', report=str(REPORT / 'report.md'))), flush=True)
        except BaseException as exc:
            failure = dict(time=utc(), error=repr(exc), traceback=traceback.format_exc())
            atomic_json(RUN / f'failure_{time.time_ns()}.json', failure)
            atomic_json(RUN / 'status.json', dict(status='interrupted', **failure))
            raise


def make_report(manifest, data):
    """Read-only model inputs; creates only new v2 reports."""
    spec, base = manifest['identity']['spec'], manifest['identity']['base_protocol']
    old = collect_predictions(DEFAULT_RUN, data, base)
    predictions = dict(v1=old['swin'], logistic=old['logistic'])
    selections = read_json(RUN / 'selections.json')
    prediction_rows, fit_rows, histories = [], [], []
    for candidate in spec['candidates']:
        arm = candidate['id']
        predictions[arm] = []
        for seed in spec['seeds']:
            full = np.full(len(data['y']), -1, dtype=int)
            coverage = np.zeros(len(full), dtype=int)
            for outer in range(5):
                folder = RUN / 'outer' / f'outer{outer}_{arm}_seed{seed}'
                if not verify_trial(folder):
                    raise ValueError('Outer fit incomplete')
                epoch = selections[outer]['per_candidate'][arm]['epoch']
                expected = np.flatnonzero(data['fold'] == outer)
                with np.load(folder / f'epoch_{epoch:03d}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['indices'], expected)
                    np.testing.assert_array_equal(z['patients'], data['patients'][expected])
                    np.testing.assert_array_equal(z['y'], data['y'][expected])
                    np.testing.assert_array_equal(z['pred'], z['probabilities'].argmax(1))
                    full[expected] = z['pred']
                    coverage[expected] += 1
                    for j, index in enumerate(expected):
                        prediction_rows.append(dict(arm=arm, seed=seed, outer_fold=outer, epoch=epoch,
                            window_id=int(index), patient=data['patients'][index], row_id=int(data['row_ids'][index]),
                            window_start=int(data['window_start'][index]), label=base['classes'][data['y'][index]],
                            prediction=base['classes'][z['pred'][j]],
                            **{f'p_{c}':float(z['probabilities'][j,k]) for k,c in enumerate(base['classes'])}))
            if not np.all(coverage == 1):
                raise ValueError('Outer coverage invalid')
            predictions[arm].append(full)
    selected = []
    for i, seed in enumerate(spec['seeds']):
        full = np.empty(len(data['y']), dtype=int)
        for outer in range(5):
            mask = data['fold'] == outer
            full[mask] = predictions[selections[outer]['selected']['candidate']][i][mask]
        selected.append(full)
    predictions['swin'] = selected  # Shared bootstrap helper names the primary arm 'swin'.
    selected_rows = [dict(row, arm='selected') for row in prediction_rows
                     if row['arm'] == selections[row['outer_fold']]['selected']['candidate']]
    prediction_rows += selected_rows
    for kind in ('inner', 'outer'):
        for marker in sorted((RUN / kind).glob('*/completed.json')):
            folder = marker.parent
            if not verify_trial(folder):
                raise ValueError('Fit incomplete during report')
            record = read_json(folder / 'fit.json')
            identity = read_json(folder / 'identity.json')
            fit_rows.append(dict(trial=folder.name, kind=kind, seed=record['seed'], dropout=record['dropout'],
                epochs=record['epochs'], seconds=record['seconds'], reused_v1=record.get('reused_v1', False), train_patients=record['train_patients'],
                evaluation_patients=record['evaluation_patients'], parameter_count=record['parameter_count'],
                train_accuracy=record['training']['accuracy'], train_macro_f1=record['training']['macro_f1'],
                max_reload_error=max(x['maximum_probability_error'] for x in record['reload_checks'])))
            for row in read_json(folder / 'history.json'):
                histories.append(dict(trial=folder.name, kind=kind, seed=record['seed'],
                    dropout=record['dropout'], epoch=row['epoch'], loss=row['loss'], lr=row['lr'],
                    evaluation_accuracy=row.get('evaluation_accuracy', ''), evaluation_macro_f1=row.get('evaluation_macro_f1', '')))
    results, metrics = {}, []
    for arm, items in predictions.items():
        per_seed = []
        for i, prediction in enumerate(items):
            window, patient = patient_confusions(data['y'], prediction, data['patients'])
            values = dict(window=scores(window.sum(0)), patient_vote=scores(patient.sum(0)))
            per_seed.append(values)
            metrics.append(dict(arm='selected' if arm == 'swin' else arm,
                seed='' if arm == 'logistic' else spec['seeds'][i],
                **{f'{scope}_{key}':values[scope][key] for scope in ('window', 'patient_vote')
                   for key in ('accuracy', 'macro_f1', 'balanced_accuracy')},
                **{f'window_recall_{c}':values['window']['sensitivity'][k] for k,c in enumerate(base['classes'])}))
        results[arm] = dict(per_seed=per_seed, mean={scope:{key:float(np.mean([v[scope][key] for v in per_seed]))
                           for key in ('accuracy', 'macro_f1', 'balanced_accuracy')} for scope in ('window', 'patient_vote')})
    bootstrap = bootstrap_comparison(data['y'], data['patients'], predictions,
                                      spec['bootstrap_resamples'], spec['bootstrap_seed'])
    secondary = {}
    for arm in ('R0', 'R1'):
        secondary[arm] = bootstrap_comparison(data['y'], data['patients'],
            {'swin':predictions[arm], 'v1':predictions['v1']}, spec['bootstrap_resamples'], spec['bootstrap_seed'])
    analysis = dict(results=results, selected_arm_key='swin', seed_order=spec['seeds'], selections=selections,
                    bootstrap=bootstrap, secondary_bootstrap=secondary,
                    uncertainty_scope='Pointwise patient-bootstrap intervals conditional on predictions; exclude retraining and selection uncertainty.',
                    fit_seconds=sum(r['seconds'] for r in fit_rows),
                    reused_outer_fits=sum(r['reused_v1'] for r in fit_rows),
                    failures=[p.name for p in RUN.glob('failure_*.json')],
                    protocol=spec, manifest_sha256=file_hash(RUN / 'manifest.json'))
    atomic_json(REPORT / 'analysis.json', analysis)
    write_csv(REPORT / 'metrics.csv', metrics)
    write_csv(REPORT / 'predictions.csv', prediction_rows)
    write_csv(REPORT / 'fits.csv', fit_rows)
    write_csv(REPORT / 'histories.csv', histories)
    write_csv(REPORT / 'selection.csv', [dict(outer_fold=s['outer_fold'], candidate=r['candidate'], epoch=r['epoch'],
        mean_macro_f1=r['mean_macro_f1'], **{f'seed{seed}_macro_f1':r['seed_macro_f1'][i] for i,seed in enumerate(spec['seeds'])})
        for s in selections for r in s['rows']])
    for name in ('analysis.json', 'metrics.csv', 'predictions.csv', 'fits.csv', 'histories.csv', 'selection.csv', 'splits.csv'):
        (RUN / name).write_bytes((REPORT / name).read_bytes())
    interval = bootstrap['paired_difference_intervals_95']['swin_minus_v1']['window_macro_f1']
    delta = results['swin']['mean']['window']['macro_f1'] - results['v1']['mean']['window']['macro_f1']
    text = [
        '# ECGData LFCC–Swin regularization v2 — completed', '',
        f'Completed {utc()}. All 90 inner fits and 30 outer trials completed; {analysis["reused_outer_fits"]} outer trials reused saved v1 fits. Seeds: 0, 1, 2.', '',
        f'Primary selected-procedure minus v1 window macro-F1: **{delta:+.4f}**, paired 95% interval **[{interval[0]:+.4f}, {interval[1]:+.4f}]**.', '',
        ('The interval supports improvement within this exploratory cohort.' if interval[0] > 0 else
         'This experiment does not establish an improvement over v1.'), '',
        '| Pipeline | Window accuracy | Window macro-F1 | Patient-vote accuracy |',
        '|---|---:|---:|---:|']
    labels = {'swin':'Inner-selected procedure', 'R0':'R0: epoch selection only',
              'R1':'R1: dropout 0.1 + epoch selection', 'v1':'Saved v1: fixed epoch 40', 'logistic':'Saved pooled LFCC logistic'}
    for arm in ('swin', 'R0', 'R1', 'v1', 'logistic'):
        row = results[arm]['mean']
        text.append(f'| {labels[arm]} | {100*row["window"]["accuracy"]:.2f}% | {row["window"]["macro_f1"]:.4f} | {100*row["patient_vote"]["accuracy"]:.2f}% |')
    text += ['', 'Stochastic rows average all three seeds; they are not seed ensembles.', '',
             '## Selected settings', '', '| Outer fold | R0 epoch | R1 epoch | Joint selection |', '|---|---:|---:|---|']
    for s in selections:
        text.append(f'| {s["outer_fold"]} | {s["per_candidate"]["R0"]["epoch"]} | {s["per_candidate"]["R1"]["epoch"]} | {s["selected"]["candidate"]}, epoch {s["selected"]["epoch"]} |')
    text += ['', '## Every seed', '', '| Pipeline | Seed | Window accuracy | Macro-F1 | Patient-vote accuracy |', '|---|---:|---:|---:|---:|']
    for row in metrics:
        text.append(f'| {row["arm"]} | {row["seed"]} | {100*row["window_accuracy"]:.2f}% | {row["window_macro_f1"]:.4f} | {100*row["patient_vote_accuracy"]:.2f}% |')
    text += ['', '## Evaluation and limitations', '',
        'Same 1,620 windows and 80 patients as v1; no exclusions. Five outer patient folds, three stratified inner patient folds; fit-only normalization and class weights. All candidate selections were saved before outer evaluation. Earlier refits retain the 40-epoch cosine horizon.', '',
        'Primary outcome is window macro-F1. All intervals use 2,000 paired patient draws (seed 20260925), average initialization seeds inside draws, and are pointwise without multiplicity adjustment. They condition on predictions and exclude retraining/selection uncertainty.', '',
        'Repeatedly studied 80-patient cohort; diagnosis/source confounding remains. This is exploratory internal evidence, not external or clinical validation. Patient voting aggregates multiple windows/leads and is a different prediction unit. No advantage of a particular transform or quantum model is established.', '',
        f'Sum of fit runtimes: {analysis["fit_seconds"]:.3f} seconds, including fit-level checkpointing and reload checks; excludes global preflight/reporting. Failed invocation records: {len(analysis["failures"])}. All retained candidate and final weights were reloaded on the training device with saved probabilities and predicted classes checked.', '',
        '## Artifacts', '',
        '- [Full analysis, per-class metrics, confusion matrices and intervals](analysis.json)',
        '- [Every seed](metrics.csv), [outer predictions](predictions.csv), [fits and runtimes](fits.csv)',
        '- [Training and inner-validation histories](histories.csv), [all candidate scores](selection.csv), [patient splits](splits.csv)',
        '- [Protocol](../../ecgdata_regularization_protocol_v2.md)', '',
        'Per-fit inner validation probabilities at epochs 10/20/40, retained weights, exact window indices, optimizer/RNG checkpoints, source snapshots and integrity markers are under the matching results directory (gitignored). Saved v1 models and reports are unchanged.', '']
    (REPORT / 'report.md').write_text('\n'.join(text))
    (RUN / 'report.md').write_bytes((REPORT / 'report.md').read_bytes())
