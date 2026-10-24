"""Balanced fits with deterministic epoch recovery and retained candidate scores."""
import io
import os
import platform
import time
import warnings

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import common as c
from .data import normalization
from .models import model
import torch


def setup(device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must equal :4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    return dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
        device=device, device_name=torch.cuda.get_device_name() if device == 'cuda' else platform.processor(),
        deterministic=True, tf32=False, precision='float32')


def checkpoint(folder, state):
    name = f'checkpoint{state["epoch"] % 2}.pt'
    stream = io.BytesIO()
    torch.save(state, stream)
    c.write_bytes(folder/name, stream.getvalue())
    c.write(folder/'checkpoint.json', dict(file=name, sha256=c.sha(folder/name), epoch=state['epoch']))


def load_checkpoint(folder, context):
    if not (folder/'checkpoint.json').exists():
        return None
    pointer = c.read(folder/'checkpoint.json')
    if c.sha(folder/pointer['file']) != pointer['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    value = torch.load(folder/pointer['file'], map_location='cpu', weights_only=True)
    if value['context'] != context or value['epoch'] != pointer['epoch']:
        raise ValueError('Checkpoint context mismatch')
    return value


def batch(X, ix, mean, scale, device):
    return torch.from_numpy(((np.asarray(X[ix], dtype=np.float64)-mean)/scale).astype(np.float32)).to(device)


def predict(net, X, indices, mean, scale, batch_size, device):
    net.eval()
    scores = []
    with torch.inference_mode():
        for first in range(0,len(indices),batch_size):
            x = batch(X,indices[first:first+batch_size],mean,scale,device)
            scores.append(net(x).softmax(-1)[:,1].cpu().numpy())
    score = np.concatenate(scores)
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError('Invalid prediction')
    return score


def predictions(path, data, indices, score):
    if np.asarray(score).shape != (len(indices),) or not np.isfinite(score).all():
        raise ValueError('Invalid saved prediction')
    c.npz(path, score=score, y=data['y'][indices], patients=data['patients'][indices],
          record_ids=data['record_ids'][indices], indices=indices)


def fit_neural(folder, data, X, sampled, held, arm, seed, epochs, spec, ctx, device,
               inner=False, stop_after=None):
    if c.verified(folder,ctx):
        return
    folder.mkdir(parents=True,exist_ok=True)
    unique = np.unique(sampled)
    if set(data['patients'][unique]) & set(data['patients'][held]):
        raise ValueError('Fit/held patient overlap')
    settings = dict(arm=arm,seed=seed,epochs=epochs,inner=inner,training=spec['training'])
    c.frozen_write(folder/'settings.json',settings)
    c.npz(folder/'indices.npz',unique=unique,sampled=sampled,held=held)
    normpath = folder/'normalization.npz'
    if normpath.exists():
        norm = c.arrays(normpath)
        np.testing.assert_array_equal(norm['fit_indices'],unique)
        mean,scale = norm['mean'],norm['scale']
    else:
        mean,scale = normalization(X,unique,raw=arm=='raw')
        c.npz(normpath,mean=mean,scale=scale,fit_indices=unique)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    net = model(arm,spec).to(device)
    cfg = spec['training']
    opt = torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=cfg['min_lr'])
    history,first,prior = [],0,0.
    state = load_checkpoint(folder,ctx)
    if state is not None:
        if state['settings'] != settings:
            raise ValueError('Checkpoint settings mismatch')
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        history,first,prior = state['history'],state['epoch'],state['seconds']
        torch.set_rng_state(state['rng'])
        if device=='cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    start=time.perf_counter()
    limit=epochs if stop_after is None else min(epochs,stop_after)
    warnings_path=folder/'warnings.json'
    previous_warnings=c.read(warnings_path) if warnings_path.exists() else []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for epoch in range(first,limit):
            net.train()
            total,count=0.,0
            tick=time.perf_counter()
            lr=opt.param_groups[0]['lr']
            order=np.random.default_rng(seed+cfg['shuffle_seed_offset']+epoch).permutation(len(sampled))
            for firstbatch in range(0,len(order),cfg['batch_size']):
                ix=sampled[order[firstbatch:firstbatch+cfg['batch_size']]]
                opt.zero_grad(set_to_none=True)
                target=torch.from_numpy(data['y'][ix].astype(np.int64)).to(device)
                loss=torch.nn.functional.cross_entropy(net(batch(X,ix,mean,scale,device)),target)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
                opt.step()
                total+=float(loss.detach().cpu())*len(ix)
                count+=len(ix)
            sched.step()
            record=dict(epoch=epoch+1,loss=total/count,lr=lr,samples=count,
                        steps=int(np.ceil(count/cfg['batch_size'])),seconds=time.perf_counter()-tick)
            if inner or epoch+1==epochs:
                score=predict(net,X,held,mean,scale,cfg['batch_size'],device)
                if inner:
                    record['inner_AP']=float(average_precision_score(data['y'][held],score))
                if (inner and epoch+1 in spec['epochs']) or (not inner and epoch+1==epochs):
                    candidate=folder/f'epoch{epoch+1}'
                    stream=io.BytesIO()
                    torch.save(net.state_dict(),stream)
                    c.write_bytes(candidate/'weights.pt',stream.getvalue())
                    # Construction consumes RNG; restore it to preserve training trajectory.
                    rng=torch.get_rng_state()
                    cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []
                    restored=model(arm,spec).to(device)
                    restored.load_state_dict(torch.load(candidate/'weights.pt',map_location=device,weights_only=True))
                    np.testing.assert_array_equal(predict(restored,X,held,mean,scale,cfg['batch_size'],device),score)
                    torch.set_rng_state(rng)
                    if device=='cuda':
                        torch.cuda.set_rng_state_all(cuda_rng)
                    del restored
                    predictions(candidate/'held.npz',data,held,score)
                    c.complete(candidate,ctx,reload_exact=True)
            history.append(record)
            checkpoint(folder,dict(epoch=epoch+1,context=ctx,settings=settings,model=net.state_dict(),
                optimizer=opt.state_dict(),scheduler=sched.state_dict(),history=history,
                seconds=prior+time.perf_counter()-start,rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []))
            c.write(folder/'history.json',history)
            c.write(folder/'status.json',dict(status='running',epoch=epoch+1,epochs=epochs,updated=c.now()))
            print(f'{folder.relative_to(c.RUN) if folder.is_relative_to(c.RUN) else folder.name}: '
                  f'epoch {epoch+1}/{epochs}, loss={record["loss"]:.5f}, {record["seconds"]:.1f}s',flush=True)
    c.write(warnings_path,previous_warnings+[str(w.message) for w in caught])
    if limit<epochs:
        return
    predictions(folder/'training.npz',data,unique,predict(net,X,unique,mean,scale,cfg['batch_size'],device))
    c.write(folder/'fit.json',dict(settings=settings,seconds=prior+time.perf_counter()-start,
        parameters=sum(p.numel() for p in net.parameters()),unique_fit_records=len(unique),
        balanced_counts=np.bincount(data['y'][sampled],minlength=2).tolist(),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        completed_at_utc=c.now()))
    c.complete(folder,ctx)
    c.write(folder/'status.json',dict(status='complete',completed_at_utc=c.now()))


def fit_logistic(folder,data,X,sampled,held,C,ctx):
    if c.verified(folder,ctx):
        return
    start=time.perf_counter()
    unique=np.unique(sampled)
    if set(data['patients'][unique]) & set(data['patients'][held]):
        raise ValueError('Logistic patient overlap')
    scaler=StandardScaler().fit(X[unique])
    clf=LogisticRegression(C=C,max_iter=2000,solver='lbfgs',random_state=0)
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
        warnings.simplefilter('always')
        clf.fit(scaler.transform(X[sampled]),data['y'][sampled])
    if clf.classes_.tolist()!=[0,1]:
        raise ValueError('Class order changed')
    stream=io.BytesIO()
    joblib.dump((scaler,clf),stream)
    c.write_bytes(folder/'model.joblib',stream.getvalue())
    restored_scaler,restored=joblib.load(folder/'model.joblib')
    score=clf.predict_proba(scaler.transform(X[held]))[:,1]
    np.testing.assert_array_equal(score,restored.predict_proba(restored_scaler.transform(X[held]))[:,1])
    predictions(folder/'held.npz',data,held,score)
    predictions(folder/'training.npz',data,unique,clf.predict_proba(scaler.transform(X[unique]))[:,1])
    c.npz(folder/'indices.npz',unique=unique,sampled=sampled,held=held)
    c.write(folder/'fit.json',dict(C=C,seconds=time.perf_counter()-start,iterations=clf.n_iter_.tolist(),
        warnings=[str(w.message) for w in caught],reload_exact=True,completed_at_utc=c.now()))
    c.complete(folder,ctx)
