"""Partition identities, normalization and resumable native-waveform cache."""
import csv
import platform
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import common as c
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.prepare import oversample


def inner_splits(data, spec):
    sub = np.flatnonzero(data['partitions'] == 'subfit')
    splitter = StratifiedGroupKFold(spec['inner']['folds'], shuffle=True,
                                    random_state=spec['inner']['seed'])
    result = []
    for train, held in splitter.split(np.zeros(len(sub)), data['y'][sub], data['patients'][sub]):
        train, held = sub[train], sub[held]
        if set(data['patients'][train]) & set(data['patients'][held]):
            raise ValueError('Inner patient overlap')
        if any(set(data['y'][ix]) != {0, 1} for ix in (train, held)):
            raise ValueError('Missing class')
        result.append((train, held))
    np.testing.assert_array_equal(np.sort(np.concatenate([v for _, v in result])), sub)
    return result


def normalization(X, indices, raw=False):
    """Stable streaming population statistics; raw is N,C,T; cepstra N,T,L,C."""
    count, mean, m2 = 0, None, None
    for first in range(0, len(indices), 32):
        x = np.asarray(X[indices[first:first+32]], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite features')
        if raw:
            x = x.transpose(0, 2, 1)
        x = x.reshape(-1, *x.shape[2:])
        n, local = len(x), x.mean(0)
        local_m2 = ((x-local)**2).sum(0)
        if mean is None:
            mean, m2 = local, local_m2
        else:
            delta = local-mean
            m2 += local_m2 + delta**2 * count*n/(count+n)
            mean += delta*n/(count+n)
        count += n
    scale = np.sqrt(m2/count)
    scale[scale < 1e-8] = 1
    if raw:
        mean, scale = mean[:, None], scale[:, None]
    return mean, scale


def prepare():
    started = time.perf_counter()
    spec = c.read(c.BASE / 'protocol.json')
    if (c.RUN / 'manifest.json').exists():
        _, ctx = c.manifest()
        if c.verified(c.RUN / 'prepared', ctx):
            print('Preparation already complete and verified', flush=True)
            return
    old = c.read(c.PARENT / 'results/manifest.json')
    protected = dict(old['protected_sha256'])
    for marker in sorted((c.PARENT / 'results').rglob('completed.json')):
        saved = c.read(marker)
        if saved['status'] != 'complete':
            raise ValueError(f'Incomplete parent {marker}')
        protected[str(marker.relative_to(c.ROOT))] = c.sha(marker)
        for name, digest in saved['files'].items():
            protected[str((marker.parent/name).relative_to(c.ROOT))] = digest
    protected[str((c.PARENT/'results/manifest.json').relative_to(c.ROOT))] = c.sha(c.PARENT/'results/manifest.json')
    c.protect_check({'protected_sha256': protected})
    # Pin only executable dependencies and protocol, not evolving prose/logs.
    paths = list(c.BASE.glob('*.py')) + list((c.BASE/'tests').glob('*.py')) + [c.BASE/'protocol.json']
    paths += [c.PARENT / name for name in ('common.py', 'prepare.py', 'metrics.py', 'fit.py')]
    paths += [c.ROOT/'attention method/attention_ecg'/name for name in ('model.py', 'regularized_swin.py')]
    paths += [c.BASE.parent/'loader.py']
    code = dict(old['code_sha256'])
    code.update({str(p.relative_to(c.ROOT)): c.sha(p) for p in paths})
    value = dict(protocol=spec, code_sha256=code, protected_sha256=protected,
                 environment={'python': platform.python_version()})
    c.frozen_write(c.RUN/'manifest.json', value)
    ctx = c.sha(c.RUN/'manifest.json')
    for name in code:
        c.write_bytes(c.RUN/'source'/name, (c.ROOT/name).read_bytes())
    out = c.RUN/'prepared'
    data = c.arrays(c.PARENT/'results/prepared/cohort.npz')
    for a in ('subfit', 'calibration', 'validation'):
        for b in ('subfit', 'calibration', 'validation'):
            if a != b and set(data['patients'][data['partitions'] == a]) & set(data['patients'][data['partitions'] == b]):
                raise ValueError('Parent patient overlap')
    if set(data['partitions']) != {'subfit', 'calibration', 'validation'}:
        raise ValueError('Unexpected cohort')
    c.npz(out/'cohort.npz', **data)
    oldlfcc = c.arrays(c.BALANCED/'results/v1/lfcc/features.npz')
    lookup = {rid: i for i, rid in enumerate(oldlfcc['record_ids'])}
    if set(lookup) != set(data['record_ids']):
        raise ValueError('LFCC cohort differs')
    order = np.array([lookup[rid] for rid in data['record_ids']])
    for key in ('patients', 'y'):
        np.testing.assert_array_equal(oldlfcc[key][order], data[key])
    X = oldlfcc['X'][order]
    pooled = np.concatenate([X.mean(1).reshape(len(X), -1), X.std(1).reshape(len(X), -1)], axis=1)
    for name, array in [('lfcc.npy', X), ('pooled.npy', pooled)]:
        target = np.lib.format.open_memmap(out/name, mode='w+', dtype=array.dtype, shape=array.shape)
        target[:] = array
        target.flush()
        del target
    splits = inner_splits(data, spec)
    fold_ids = np.full(len(X), -1)
    for fold, (train, held) in enumerate(splits):
        sampled = train[oversample(data['y'][train], spec['sampling']['inner_seed']+fold)]
        c.npz(out/f'inner{fold}.npz', train=train, held=held, sampled=sampled)
        fold_ids[held] = fold
    sub = np.flatnonzero(data['partitions'] == 'subfit')
    np.testing.assert_array_equal(sub[oversample(data['y'][sub], spec['sampling']['final_seed'])], data['sampled_indices'])
    c.csv_write(out/'splits.csv', ['record_id','patient_id','y','partition','inner_held_fold'],
                zip(data['record_ids'], data['patients'], data['y'], data['partitions'], fold_ids))
    c.write(out/'audit.json', dict(seconds=time.perf_counter()-started, records=len(X),
        partitions={p: {'records': int(np.sum(data['partitions']==p)),
                       'patients': len(set(data['patients'][data['partitions']==p])),
                       'positive': int(data['y'][data['partitions']==p].sum())}
                    for p in ('subfit','calibration','validation')},
        patient_overlap=0, new_exclusions=0, protected_files=len(protected), official_test_processed=False))
    c.complete(out, ctx)
    print('Prepared LFCC cache and three inner patient folds', flush=True)


def prepare_raw(ctx):
    from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset, LEADS
    out = c.RUN/'raw'
    if c.verified(out, ctx):
        return
    data = c.cohort()
    with (c.BASE.parent/'results/omi_v1/splits.csv').open(newline='') as stream:
        rows = {r['record_id']: r for r in csv.DictReader(stream)}
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with ACSDataset(target='OMI') as dataset:
        for first in range(0, len(data['y']), 256):
            folder = out/'batches'/f'{first:05d}'
            last = min(first+256, len(data['y']))
            if c.verified(folder, ctx):
                continue
            x, records = [], []
            for i in range(first, last):
                rid = str(data['record_ids'][i])
                rec = dataset.inspect(rid)
                row = rows[rid]
                if (rec.info.split != 'train' or rec.info.patient_id != data['patients'][i]
                    or rec.info.label != data['y'][i] or not rec.decision.eligible
                    or rec.decision.waveform_sha256 != row['waveform_sha256']
                    or rec.signal_mV.shape != (5000,12) or tuple(rec.leads) != LEADS or rec.fs != 500):
                    raise ValueError(f'Raw input identity/quality differs: {rid}')
                x.append(rec.signal_mV.T.astype(np.float32))
                records.append(rec.decision.to_dict())
            c.npz(folder/'signals.npz', X=np.stack(x), record_ids=data['record_ids'][first:last])
            c.write(folder/'records.json', records)
            c.complete(folder, ctx)
            print(f'Raw cache {last}/{len(data["y"])}', flush=True)
        archives = dataset.source_manifest
    target = np.lib.format.open_memmap(out/'signals.npy', mode='w+', dtype='float32', shape=(len(data['y']),12,5000))
    for first in range(0, len(data['y']), 256):
        batch = c.arrays(out/'batches'/f'{first:05d}'/'signals.npz')
        np.testing.assert_array_equal(batch['record_ids'], data['record_ids'][first:first+len(batch['X'])])
        target[first:first+len(batch['X'])] = batch['X']
    target.flush()
    del target
    c.write(out/'audit.json', dict(seconds_this_session=time.perf_counter()-start, archives=archives,
        records=len(data['y']), shape=[len(data['y']),12,5000], official_test_processed=False, exclusions_added=0))
    c.complete(out, ctx)
