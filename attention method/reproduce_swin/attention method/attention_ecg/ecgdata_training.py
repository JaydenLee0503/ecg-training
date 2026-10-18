"""Explicit-only ECGData fitting, epoch checkpoints, and artifact verification.

Importing this module or preparing/dry-running the experiment never fits a model.
The fixed protocol has no outer-fold selection or early stopping.
"""
import os
from pathlib import Path
import platform
import time
import traceback
import warnings

import numpy as np
import sklearn

from .ecgdata import (ROOT, PROTOCOL, DEFAULT_CACHE, read_json, file_hash, atomic_json,
                      atomic_npz, safe_output, writer_lock, verify_cache, fit_normalizer, normalize)

DEFAULT_RUN = ROOT/'results/ecgdata_lfcc_swin_v1'


def require_training(enabled):
    if not enabled:
        raise ValueError('Training is disabled. An explicit --execute-training flag is required.')


def partition(data, fold):
    if fold not in range(5):
        raise ValueError('Unknown fold')
    tr, te = np.flatnonzero(data['fold'] != fold), np.flatnonzero(data['fold'] == fold)
    if set(data['patients'][tr]) & set(data['patients'][te]):
        raise ValueError('Patient leakage')
    if set(data['y'][tr]) != {0,1,2} or set(data['y'][te]) != {0,1,2}:
        raise ValueError('Missing class')
    return tr, te


def batch_indices(n, batch_size, seed, epoch, offset=10000):
    order = np.random.default_rng(seed + offset + epoch).permutation(n)
    return [order[i:i+batch_size] for i in range(0,n,batch_size)]


def verify_trial(folder):
    marker = folder/'completed.json'
    if not marker.exists():
        return False
    info = read_json(marker)
    if info['status'] != 'complete':
        raise ValueError('Invalid trial completion marker')
    for name, digest in info['files'].items():
        if file_hash(folder/name) != digest:
            raise ValueError(f'Changed completed trial artifact: {folder/name}')
    return True


def complete_trial(folder, names):
    atomic_json(folder/'completed.json', dict(status='complete', files={name:file_hash(folder/name) for name in names}))


def save_checkpoint(folder, state):
    """Checksum/epoch marker detects interrupted writes instead of trusting a file."""
    import torch
    target = folder/'checkpoint.pt'
    tmp = target.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, target)
    atomic_json(folder/'checkpoint.json', dict(epoch=state['epoch'], sha256=file_hash(target)))


def load_checkpoint(folder):
    import torch
    marker = folder/'checkpoint.json'
    target = folder/'checkpoint.pt'
    if not marker.exists():
        if target.exists():
            raise ValueError('Orphan checkpoint without checksum; inspect before restarting trial')
        return None
    info = read_json(marker)
    if file_hash(target) != info['sha256']:
        raise ValueError('Incomplete or corrupt checkpoint; inspect retained files')
    state = torch.load(target, map_location='cpu', weights_only=True)
    if state['epoch'] != info['epoch']:
        raise ValueError('Checkpoint epoch mismatch')
    return state


def pooled_features(X):
    return np.concatenate([X.mean(axis=1).reshape(len(X),-1),
                           X.std(axis=1).reshape(len(X),-1)], axis=1)


def fit_control(folder, data, fold, protocol):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if verify_trial(folder):
        return
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tr, te = partition(data, fold)
    X = pooled_features(data['X'])
    c = protocol['control']
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c['C'], class_weight=c['class_weight'],
                         max_iter=c['max_iter'], solver=c['solver'], random_state=0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(X[tr], data['y'][tr])
    scores = model.predict_proba(X[te])
    if not np.array_equal(model[-1].classes_, np.arange(3)) or not np.isfinite(scores).all():
        raise ValueError('Invalid control scores')
    tmp = folder/'model.tmp'
    joblib.dump(model, tmp)
    os.replace(tmp, folder/'model.joblib')
    atomic_npz(folder/'predictions.npz', indices=te, y=data['y'][te], probabilities=scores,
               pred=scores.argmax(1), patients=data['patients'][te])
    atomic_json(folder/'fit.json', dict(fold=fold, train_indices=tr.tolist(), test_indices=te.tolist(),
                seconds=time.perf_counter()-start, warnings=[str(w.message) for w in caught],
                iterations=model[-1].n_iter_.tolist(), classes=protocol['classes']))
    complete_trial(folder, ['model.joblib','predictions.npz','fit.json'])


def fit_swin(folder, data, fold, seed, protocol, device):
    import torch
    from .model import CepstralSwin

    if verify_trial(folder):
        return
    folder.mkdir(parents=True, exist_ok=True)
    tr, te = partition(data, fold)
    settings = protocol['training']
    mean, scale = fit_normalizer(data['X'], tr)
    X = normalize(data['X'], mean, scale)
    atomic_npz(folder/'preprocessing.npz', mean=mean, scale=scale, train_indices=tr, test_indices=te)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = CepstralSwin(**protocol['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['epochs'], eta_min=settings['min_lr'])
    counts = np.bincount(data['y'][tr], minlength=3)
    weights = len(tr)/(3*counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    history, first, prior_seconds = [], 0, 0.0
    state = load_checkpoint(folder)
    if state is not None:
        if state['seed'] != seed or state['fold'] != fold:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, first, prior_seconds = state['history'], state['epoch'], state['seconds']
        if not 0 <= first <= settings['epochs'] or len(history) != first:
            raise ValueError('Invalid checkpoint history')
        torch.set_rng_state(state['rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(first, settings['epochs']):
        model.train()
        losses, batch_weights = [], []
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        for batch in batch_indices(len(tr), settings['batch_size'], seed, epoch, settings['shuffle_seed_offset']):
            idx = tr[batch]
            inputs = torch.from_numpy(X[idx]).to(device)
            target = torch.from_numpy(data['y'][idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_weights.append(float(weights[data['y'][idx]].sum()))
        scheduler.step()
        history.append(dict(epoch=epoch+1, loss=float(np.average(losses,weights=batch_weights)),
                            lr=lr, seconds=time.perf_counter()-epoch_start))
        save_checkpoint(folder, dict(epoch=epoch+1, seed=seed, fold=fold, history=history,
                        seconds=prior_seconds+time.perf_counter()-start, model=model.state_dict(),
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []))
        atomic_json(folder/'history.json', history)
    model.eval()
    scores = []
    with torch.inference_mode():
        for i in range(0,len(te),settings['batch_size']):
            scores.append(model(torch.from_numpy(X[te[i:i+settings['batch_size']]]).to(device)).softmax(-1).cpu().numpy())
    scores = np.concatenate(scores)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite predictions')
    atomic_npz(folder/'predictions.npz', indices=te, y=data['y'][te], probabilities=scores,
               pred=scores.argmax(1), patients=data['patients'][te])
    atomic_json(folder/'fit.json', dict(seed=seed, fold=fold, classes=protocol['classes'],
                model_config=protocol['model'], class_weights=weights.tolist(),
                train_indices=tr.tolist(), test_indices=te.tolist(), epochs=settings['epochs'],
                seconds=prior_seconds+time.perf_counter()-start,
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None))
    # Restore the human-readable history even after interruption following checkpoint commit.
    atomic_json(folder/'history.json', history)
    complete_trial(folder, ['checkpoint.pt','checkpoint.json','preprocessing.npz',
                           'history.json','predictions.npz','fit.json'])


def run_training(cache=DEFAULT_CACHE, output=DEFAULT_RUN, *, execute=False, device='cpu'):
    require_training(execute)  # Must precede directories, imports and fitting.
    import torch
    from . import model as model_module

    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable; no silent CPU fallback')
    output = safe_output(output, 'results')
    cache_manifest, data = verify_cache(cache)
    protocol = cache_manifest['protocol']
    torch.set_num_threads(protocol['training']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    if device=='cuda':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA training')
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    manifest = dict(protocol_sha256=file_hash(PROTOCOL), cache_manifest_sha256=file_hash(Path(cache)/'manifest.json'),
                    cache_completed_sha256=file_hash(Path(cache)/'completed.json'),
                    sources={str(p.relative_to(ROOT)):file_hash(p) for p in
                             (Path(__file__), Path(model_module.__file__), ROOT/'ecgdata.py')},
                    versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                                  torch=torch.__version__, cuda=torch.version.cuda), device=device,
                    device_name=torch.cuda.get_device_name() if device=='cuda' else platform.processor(),
                    protocol=protocol)
    with writer_lock(output):
        if (output/'manifest.json').exists():
            if read_json(output/'manifest.json') != manifest:
                raise ValueError('Training manifest differs; use a new run directory')
        else:
            atomic_json(output/'manifest.json', manifest)
        if (output/'completed.json').exists():
            if not verify_trial(output):
                raise ValueError('Invalid completed run')
            for fold in range(5):
                if not verify_trial(output/f'logistic_fold{fold}'):
                    raise ValueError('Missing control trial')
                for seed in protocol['training']['seeds']:
                    if not verify_trial(output/f'swin_seed{seed}_fold{fold}'):
                        raise ValueError('Missing Swin trial')
            return dict(status='reused', training_started=False)
        atomic_json(output/'status.json', dict(status='running'))
        try:
            for fold in range(5):
                fit_control(output/f'logistic_fold{fold}', data, fold, protocol)
                for seed in protocol['training']['seeds']:
                    fit_swin(output/f'swin_seed{seed}_fold{fold}', data, fold, seed, protocol, device)
            complete_trial(output, ['manifest.json'] +
                           [str(p.relative_to(output)) for p in sorted(output.glob('*/completed.json'))])
            atomic_json(output/'status.json', dict(status='complete', trials=20))
        except BaseException as exc:
            atomic_json(output/f'failure_{time.time_ns()}.json', dict(type=type(exc).__name__,
                        message=str(exc), traceback=traceback.format_exc()))
            atomic_json(output/'status.json', dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'))
            raise
    return dict(status='complete', trials=20)
