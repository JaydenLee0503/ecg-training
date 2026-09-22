#!/usr/bin/env python
"""Recover ECGData row identities by matching public PhysioNet source excerpts.

Reference files are identity evidence only; training continues to use ECGData.mat.
The verifier needs 12-second format-212 excerpts at 0 and 256 seconds plus each
source header in --cache. It compares the interior ten seconds after resampling.
Both excerpts must independently identify the same row with correlation >= .999,
and every row/source channel must occur exactly once. No row-order assumptions.

Records mitdb/201 and mitdb/202 belong to the same subject, as documented at
https://physionet.org/physiobank/database/html/mitdbdir/intro.htm .
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = {'mitdb': ('ARR', 360, 48), 'chfdb': ('CHF', 250, 15),
           'nsrdb': ('NSR', 128, 18)}
TIMES = (0, 256)


def row_hash(row):
    return hashlib.sha256(np.asarray(row, dtype='<f8').tobytes()).hexdigest()


def decode_212(raw):
    """Two signed 12-bit samples per byte triplet (WFDB SIGNAL(5))."""
    b = np.frombuffer(raw, dtype=np.uint8).astype(np.int32).reshape(-1, 3)
    samples = np.column_stack([b[:, 0] + ((b[:, 1] & 15) << 8),
                               b[:, 2] + ((b[:, 1] >> 4) << 8)])
    samples[samples >= 2048] -= 4096
    return samples


def normalized(x):
    x = x - x.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError('Constant signal cannot establish source identity')
    return x / norm


def verify(mat, cache):
    raw = loadmat(mat)['ECGData']
    x = np.asarray(raw['Data'][0, 0], dtype=np.float64)
    labels = np.asarray([str(np.asarray(v[0]).ravel()[0])
                         for v in raw['Labels'][0, 0]])
    if x.shape != (162, 65536):
        raise ValueError(f'Unexpected ECGData shape {x.shape}')
    targets = {t: normalized(x[:, (t+1)*128:(t+11)*128]) for t in TIMES}
    evidence = []
    for db, (label, fs, count) in SOURCES.items():
        records = (cache / f'{db}_RECORDS').read_text().split()
        if len(records) != count or len(set(records)) != count:
            raise ValueError(f'Unexpected record list for {db}')
        for record in records:
            header = (cache / f'{db}_{record}.hea').read_text().splitlines()
            first = header[0].split()
            if first[:3] != [record, '2', str(fs)]:
                raise ValueError(f'Unexpected source header: {header[0]}')
            channels = [line.split() for line in header[1:3]]
            if any(c[0] != record+'.dat' or c[1] != '212' for c in channels):
                raise ValueError(f'Only multiplexed format 212 is supported: {db}/{record}')
            matches, correlations, margins, hashes = {}, {}, {}, {}
            for t in TIMES:
                data = (cache / f'{db}_{record}_t{t}.dat').read_bytes()
                if len(data) != 12*fs*3:
                    raise ValueError(f'Wrong excerpt length: {db}/{record}, t={t}')
                signals = decode_212(data)
                if t == 0 and not np.array_equal(signals[0], [int(c[5]) for c in channels]):
                    raise ValueError(f'Decoding disagrees with header initial samples: {db}/{record}')
                resampled = resample_poly(signals, 128, fs, axis=0)[128:11*128].T
                corr = targets[t] @ normalized(resampled).T
                ranked = np.argsort(-corr, axis=0)
                matches[t] = ranked[0]
                correlations[t] = corr[ranked[0], np.arange(2)]
                margins[t] = correlations[t] - corr[ranked[1], np.arange(2)]
                hashes[t] = hashlib.sha256(data).hexdigest()
            if not np.array_equal(matches[0], matches[256]):
                raise ValueError(f'Two time excerpts disagree on row identity: {db}/{record}')
            for channel in range(2):
                row = int(matches[0][channel])
                if labels[row] != label:
                    raise ValueError(f'Class/source mismatch on row {row}')
                if min(correlations[t][channel] for t in TIMES) < .999:
                    raise ValueError(f'Weak source match on row {row}: {db}/{record}/{channel}')
                if min(margins[t][channel] for t in TIMES) < .001:
                    raise ValueError(f'Ambiguous source match on row {row}')
                patient = '201-202' if db == 'mitdb' and record in ('201', '202') else record
                entry = dict(row=row, label=label, database=db, source_record=record,
                             channel=channel, lead=channels[channel][-1],
                             source_id=f'{db}/{record}', patient_id=f'{db}/{patient}',
                             row_sha256=row_hash(x[row]),
                             source_url=f'https://physionet.org/files/{db}/1.0.0/{record}.dat')
                for t in TIMES:
                    entry[f'corr_t{t}'] = float(correlations[t][channel])
                    entry[f'margin_t{t}'] = float(margins[t][channel])
                    entry[f'excerpt_t{t}_sha256'] = hashes[t]
                evidence.append(entry)
    df = pd.DataFrame(evidence).sort_values('row').reset_index(drop=True)
    if not np.array_equal(df.row, np.arange(162)):
        raise ValueError('Source matching must cover each ECGData row exactly once')
    if not (df.groupby('source_id').size() == 2).all() or df.patient_id.nunique() != 80:
        raise ValueError('Expected 81 two-lead records and 80 patients')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mat', type=Path, default=ROOT / 'ECGData.mat')
    ap.add_argument('--cache', type=Path, default=ROOT / 'cache/physionet_sources')
    ap.add_argument('--out', type=Path, default=ROOT / 'ecgvmd/ecgdata_subjects.csv')
    args = ap.parse_args()
    df = verify(args.mat, args.cache)
    df.to_csv(args.out, index=False)
    manifest = dict(dataset_sha256=hashlib.sha256(args.mat.read_bytes()).hexdigest(),
                    rows=len(df), sources=df.source_id.nunique(), patients=df.patient_id.nunique(),
                    minimum_correlation=float(df[['corr_t0','corr_t256']].min().min()),
                    minimum_margin=float(df[['margin_t0','margin_t256']].min().min()),
                    excerpt_seconds=12, excerpt_start_seconds=list(TIMES),
                    validation='Both 10-second interiors independently identify the same source channel',
                    repeat_subject_source='https://physionet.org/physiobank/database/html/mitdbdir/intro.htm',
                    signal_format_source='https://physionet.org/physiotools/wag/signal-5.htm')
    args.out.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
