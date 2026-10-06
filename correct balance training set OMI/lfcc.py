"""Resumable ACS LFCC extraction, preserving frozen patient/record identities."""
import time
import common as c
import numpy as np
from attention_ecg.cepstral import CepstralConfig
from attention_ecg.data import extract_acs_record, load_splits, FitStandardizer
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset


def engineering_check():
    """Check one original fit record per class, without fitting or validation scoring."""
    spec, ctx, source, _, _ = c.context()
    out = c.RUN / 'lfcc_engineering'
    if c.verified(out, ctx):
        return
    rows, _ = cohort(spec, source)
    selected = [min(rid for rid, row in rows.items() if row['partition'] == 'fit' and int(row['label']) == y)
                for y in (0, 1)]
    records = []
    started = time.perf_counter()
    with ACSDataset(target='OMI') as dataset:
        for rid in selected:
            record = dataset.inspect(rid)
            x, meta = extract_acs_record(record, rows, CepstralConfig(**spec['cepstral']))
            repeated, _ = extract_acs_record(record, rows, CepstralConfig(**spec['cepstral']))
            np.testing.assert_array_equal(x, repeated)
            if list(x.shape) != spec['extraction']['shape'] or not np.isfinite(x).all():
                raise ValueError('LFCC engineering check failed')
            records.append(dict(record_id=rid, label=record.info.label, shape=list(x.shape),
                                finite=True, exact_repeat=True, waveform_sha256=meta['waveform_sha256']))
    c.write(out / 'check.json', dict(status='complete', records=records, seconds=time.perf_counter()-started,
                                    models_fitted=0, official_test_processed=False, completed_at_utc=c.now()))
    c.complete(out, ['check.json'], ctx)
    print('LFCC engineering check passed on one fit ECG per class', flush=True)


def cohort(spec, source):
    rows = load_splits(c.ROOT / spec['split'], expected_sha256=spec['split_sha256'])
    data = c.arrays(source / 'vmd/inputs.npz')
    c.validate_inputs(data)
    ids = np.concatenate([data['fit_record_ids'], data['validation_record_ids']])
    patients = np.concatenate([data['fit_patients'], data['validation_patients']])
    y = np.concatenate([data['y_fit'], data['y_validation']])
    partitions = np.array(['fit'] * len(data['y_fit']) + ['validation'] * len(data['y_validation']))
    if set(ids) != set(rows):
        raise ValueError('LFCC cohort differs from frozen VMD/WST split')
    for rid, pid, label, part in zip(ids, patients, y, partitions):
        row = rows[str(rid)]
        if row['patient_id'] != pid or int(row['label']) != label or row['partition'] != part:
            raise ValueError('LFCC cohort identity mismatch')
    return rows, dict(record_ids=ids, patients=patients, y=y, partitions=partitions)


def extract():
    spec, ctx, source, _, _ = c.context()
    engineering_check()
    out = c.RUN / 'lfcc'
    if c.verified(out, ctx):
        print('Verified completed LFCC cache', flush=True)
        return
    rows, data = cohort(spec, source)
    config = CepstralConfig(**spec['cepstral'])
    batch = spec['extraction']['batch_records']
    shape = tuple(spec['extraction']['shape'])
    out.mkdir(exist_ok=True)
    start = time.perf_counter()
    files = []
    with ACSDataset(target='OMI') as dataset:
        for first in range(0, len(data['y']), batch):
            last = min(first+batch, len(data['y']))
            folder = out / 'batches' / f'{first:05d}'
            if c.verified(folder, ctx):
                saved = c.arrays(folder / 'features.npz')
            else:
                features = []
                records = []
                t = time.perf_counter()
                for rid in data['record_ids'][first:last]:
                    record = dataset.inspect(str(rid))
                    x, meta = extract_acs_record(record, rows, config)
                    if x.shape != shape:
                        raise ValueError('Unexpected LFCC shape')
                    features.append(x)
                    records.append(meta)
                c.npz(folder / 'features.npz', X=np.stack(features), record_ids=data['record_ids'][first:last])
                c.write(folder / 'records.json', records)
                c.complete(folder, ['features.npz', 'records.json'], ctx, first=first, last=last,
                           seconds=time.perf_counter()-t)
                saved = c.arrays(folder / 'features.npz')
            np.testing.assert_array_equal(saved['record_ids'], data['record_ids'][first:last])
            if saved['X'].shape != (last-first, *shape) or not np.isfinite(saved['X']).all():
                raise ValueError('Invalid LFCC batch')
            files.append(folder / 'features.npz')
            c.write(out / 'status.json', dict(status='extracting', completed=last, total=len(data['y']),
                    seconds_this_session=time.perf_counter()-start, updated_at_utc=c.now()))
            print(f'LFCC: {last}/{len(data["y"])}', flush=True)
    X = np.concatenate([c.arrays(path)['X'] for path in files])
    fit = np.flatnonzero(data['partitions'] == 'fit')
    normalizer = FitStandardizer().fit(((str(data['record_ids'][i]), X[i]) for i in fit), rows)
    c.npz(out / 'features.npz', X=X, **data)
    c.npz(out / 'normalization.npz', mean=normalizer.mean_, scale=normalizer.scale_, fit_indices=fit,
          fit_record_ids=data['record_ids'][fit], frame_count=normalizer.frame_count_)
    c.write(out / 'normalization.json', normalizer.state_dict())
    c.write(out / 'extraction.json', dict(status='complete', records=len(X), shape=list(X.shape),
        seconds_this_session=time.perf_counter()-start, source_archives=dataset.source_manifest,
        official_test_processed=False, completed_at_utc=c.now()))
    names = ['features.npz', 'normalization.npz', 'normalization.json', 'extraction.json']
    names += [str(p.relative_to(out)) for p in sorted((out / 'batches').glob('*/*')) if p.is_file()]
    c.complete(out, names, ctx)
    c.write(out / 'status.json', dict(status='complete', records=len(X), updated_at_utc=c.now()))


def load():
    _, ctx, _, _, _ = c.context()
    out = c.RUN / 'lfcc'
    if not c.verified(out, ctx) or not c.verified(c.RUN / 'sampling', ctx):
        raise ValueError('Complete LFCC cache and sampling ledger required')
    data = c.arrays(out / 'features.npz')
    norm = c.arrays(out / 'normalization.npz')
    fit = np.flatnonzero(data['partitions'] == 'fit')
    val = np.flatnonzero(data['partitions'] == 'validation')
    if set(data['patients'][fit]) & set(data['patients'][val]):
        raise ValueError('Patient leakage')
    sampling = c.arrays(c.RUN / 'sampling/indices.npz')
    np.testing.assert_array_equal(data['record_ids'][fit], sampling['original_record_ids'])
    np.testing.assert_array_equal(data['y'][fit], sampling['original_y'])
    np.testing.assert_array_equal(data['patients'][fit], sampling['original_patients'])
    np.testing.assert_array_equal(norm['fit_indices'], fit)
    return data, norm, fit[sampling['indices']], val
