"""Fixed balanced LFCC controls and temporal Swin, with recoverable epoch commits."""
import io
import os
import platform
import time
import warnings
import common as c
import numpy as np


def setup(device):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=device,
                device_name=torch.cuda.get_device_name() if device == 'cuda' else platform.processor(),
                deterministic=True, precision='float32', tf32=False)


def save_checkpoint(folder, state):
    """Two slots keep the last committed epoch valid if a new save is interrupted."""
    import torch
    name = f'checkpoint{state["epoch"] % 2}.pt'
    stream = io.BytesIO()
    torch.save(state, stream)
    c.write_bytes(folder / name, stream.getvalue())
    c.write(folder / 'checkpoint.json', dict(file=name, sha256=c.sha(folder / name),
            epoch=state['epoch'], context_sha=state['context_sha']))


def load_checkpoint(folder, context):
    import torch
    if not (folder / 'checkpoint.json').exists():
        return None
    marker = c.read(folder / 'checkpoint.json')
    if marker['context_sha'] != context or c.sha(folder / marker['file']) != marker['sha256']:
        raise ValueError('Changed Swin checkpoint/context')
    state = torch.load(folder / marker['file'], map_location='cpu', weights_only=True)
    if state['epoch'] != marker['epoch'] or state['context_sha'] != context:
        raise ValueError('Invalid Swin checkpoint identity')
    return state


def predict(model, X, indices, batch, device):
    import torch
    scores = []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(indices), batch):
            xb = torch.from_numpy(X[indices[first:first+batch]]).to(device)
            scores.append(model(xb).softmax(-1)[:, 1].cpu().numpy())
    score = np.concatenate(scores)
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError('Invalid Swin probabilities')
    return score


def save_predictions(folder, name, data, idx, score):
    c.npz(folder / name, score=score, prediction=(score >= 0.5).astype(int),
          y=data['y'][idx], record_ids=data['record_ids'][idx], patients=data['patients'][idx])


def fit_swin(folder, data, X, tr, val, seed, spec, device, context, stop_after=None):
    import torch
    from attention_ecg.model import CepstralSwin
    if c.verified(folder, context):
        return
    folder.mkdir(parents=True, exist_ok=True)
    settings = spec['training']
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = CepstralSwin(**spec['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['epochs'],
                                                         eta_min=settings['min_lr'])
    criterion = torch.nn.CrossEntropyLoss()
    history, first, prior_seconds = [], 0, 0.0
    state = load_checkpoint(folder, context)
    if state is not None:
        if state['seed'] != seed or state['settings'] != settings or state['model_config'] != spec['model']:
            raise ValueError('Changed Swin seed/settings')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, first, prior_seconds = state['history'], state['epoch'], state['seconds']
        if not 0 <= first <= settings['epochs'] or len(history) != first:
            raise ValueError('Invalid Swin checkpoint history')
        torch.set_rng_state(state['rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    limit = min(settings['epochs'], stop_after) if stop_after is not None else settings['epochs']
    for epoch in range(first, limit):
        model.train()
        total_loss, total_samples = 0.0, 0
        lr = optimizer.param_groups[0]['lr']
        epoch_start = time.perf_counter()
        order = np.random.default_rng(seed + settings['shuffle_seed_offset'] + epoch).permutation(len(tr))
        for i in range(0, len(tr), settings['batch_size']):
            idx = tr[order[i:i+settings['batch_size']]]
            inputs = torch.from_numpy(X[idx]).to(device)
            target = torch.from_numpy(data['y'][idx].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite Swin loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            total_samples += len(idx)
        scheduler.step()
        history.append(dict(epoch=epoch+1, loss=total_loss/total_samples, lr=lr,
                            samples=total_samples, steps=int(np.ceil(len(tr)/settings['batch_size'])),
                            seconds=time.perf_counter()-epoch_start))
        save_checkpoint(folder, dict(epoch=epoch+1, seed=seed, context_sha=context, settings=settings,
                        model_config=spec['model'], history=history, seconds=prior_seconds+time.perf_counter()-start,
                        model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else []))
        c.write(folder / 'history.json', history)
        c.write(folder / 'status.json', dict(status='running', epoch=epoch+1, epochs=settings['epochs'],
                                           updated_at_utc=c.now()))
        print(f'Swin seed {seed}: epoch {epoch+1}/{settings["epochs"]}, loss={history[-1]["loss"]:.5f}, '
              f'{history[-1]["seconds"]:.1f}s', flush=True)
    if limit < settings['epochs']:
        return
    score = predict(model, X, val, settings['batch_size'], device)
    restored = CepstralSwin(**spec['model']).to(device)
    restored.load_state_dict(load_checkpoint(folder, context)['model'])
    np.testing.assert_array_equal(predict(restored, X, val, settings['batch_size'], device), score)
    save_predictions(folder, 'predictions.npz', data, val, score)
    unique_tr = np.unique(tr)
    save_predictions(folder, 'training_predictions.npz', data, unique_tr,
                     predict(model, X, unique_tr, settings['batch_size'], device))
    c.write(folder / 'history.json', history)
    c.write(folder / 'fit.json', dict(seed=seed, epochs=settings['epochs'], model_config=spec['model'],
        parameters=sum(p.numel() for p in model.parameters()), class_weight=None,
        balanced_examples=len(tr), unique_fit_records=len(unique_tr), validation_records=len(val),
        seconds=prior_seconds+time.perf_counter()-start, warnings=[],
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        reload_predictions_exact=True, completed_at_utc=c.now()))
    c.complete(folder, [p.name for p in folder.glob('checkpoint*')] +
               ['history.json', 'fit.json', 'predictions.npz', 'training_predictions.npz'], context)
    c.write(folder / 'status.json', dict(status='complete', epoch=settings['epochs'], updated_at_utc=c.now()))


def pooled(X):
    return np.concatenate([X.mean(1).reshape(len(X), -1), X.std(1).reshape(len(X), -1)], axis=1)


def fit_control(folder, data, tr, val, spec, context):
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    if c.verified(folder, context):
        return
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    X = pooled(data['X'])
    scaler = StandardScaler().fit(X[np.unique(tr)])
    params = {k: spec['control'][k] for k in ('C', 'class_weight', 'max_iter', 'solver')}
    classifier = LogisticRegression(**params, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        classifier.fit(scaler.transform(X[tr]), data['y'][tr])
    model = Pipeline([('scaler', scaler), ('classifier', classifier)])
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError('Unexpected class order')
    score = model.predict_proba(X[val])[:, 1]
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite LFCC control predictions')
    stream = io.BytesIO()
    joblib.dump(model, stream)
    c.write_bytes(folder / 'model.joblib', stream.getvalue())
    np.testing.assert_array_equal(joblib.load(folder / 'model.joblib').predict_proba(X[val])[:, 1], score)
    save_predictions(folder, 'predictions.npz', data, val, score)
    c.write(folder / 'fit.json', dict(seconds=time.perf_counter()-start, class_weight=None,
        balanced_examples=len(tr), original_scaler_records=len(np.unique(tr)),
        iterations=classifier.n_iter_.tolist(), warnings=[str(w.message) for w in caught],
        reload_predictions_exact=True, completed_at_utc=c.now()))
    c.complete(folder, ['model.joblib', 'predictions.npz', 'fit.json'], context)


def train(device='cuda'):
    import lfcc
    spec, ctx, _, _, _ = c.context()
    env = setup(device)
    data, norm, tr, val = lfcc.load()
    out = c.RUN / 'lfcc_models'
    out.mkdir(exist_ok=True)
    manifest = dict(context_sha=ctx, environment=env, features_completed_sha256=c.sha(c.RUN / 'lfcc/completed.json'),
                    sampling_completed_sha256=c.sha(c.RUN / 'sampling/completed.json'))
    if (out / 'manifest.json').exists() and c.read(out / 'manifest.json') != manifest:
        raise ValueError('Changed Swin environment/input context')
    if not (out / 'manifest.json').exists():
        c.write(out / 'manifest.json', manifest)
    training_sha = c.sha(out / 'manifest.json')
    fit_control(out / 'logistic', data, tr, val, spec, training_sha)
    X = ((data['X'].astype(np.float64) - norm['mean']) / norm['scale']).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite normalized LFCC')
    for seed in spec['training']['seeds']:
        fit_swin(out / f'swin_seed{seed}', data, X, tr, val, seed, spec, device, training_sha)
    c.complete(out, ['manifest.json'] + [str(p.relative_to(out)) for p in sorted(out.glob('*/completed.json'))], ctx)
