#!/usr/bin/env python3
"""Audit ACS loader eligibility and save every decision; does not train a model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.acs.loader import ACSDataset, LEADS, TARGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=EXPERIMENT / 'data')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
    parser.add_argument('--target', choices=TARGETS, default=None)
    parser.add_argument('--leads', nargs='+', choices=LEADS, default=list(LEADS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status_path = args.output / 'status.json'
    status_path.write_text(json.dumps({'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    try:
        with ACSDataset(args.data_dir, target=args.target, leads=args.leads) as ds:
            for split in ('train', 'test'):
                ids = ds.record_ids(split)
                for index, record_id in enumerate(ids, 1):
                    record = ds.inspect(record_id)
                    if record.decision.eligible:
                        if record.signal_mV.shape != (5000, len(ds.leads)) or not np.isfinite(record.signal_mV).all():
                            raise ValueError(f'{record_id}: invalid loader output')
                    if index % 2500 == 0 or index == len(ids):
                        print(f'{split}: {index}/{len(ids)} records checked', flush=True)
            report = ds.summary()
            report['header_conventions'] = dict(Counter(d.source_header_convention for d in ds.decisions.values()))
            report['exclusions'] = [d.to_dict() for d in ds.decisions.values() if not d.eligible]
            with (args.output / 'records.csv').open('w', newline='') as stream:
                columns = list(next(iter(ds.decisions.values())).to_dict())
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for decision in ds.decisions.values():
                    row = decision.to_dict()
                    for key, value in row.items():
                        if isinstance(value, tuple):
                            row[key] = json.dumps(value)
                    writer.writerow(row)
        report.update(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      duration_seconds=round(time.perf_counter() - started, 3),
                      python=sys.version, numpy=np.__version__,
                      scope='Loader and quality-policy validation only. No extraction, feature computation, fitting, or test-label evaluation.',
                      source_code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in [EXPERIMENT / 'loader.py', Path(__file__).resolve()]})
        report['records_csv_sha256'] = hashlib.sha256((args.output / 'records.csv').read_bytes()).hexdigest()
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        status_path.write_text(json.dumps({'status': 'complete', 'completed_at_utc': report['completed_at_utc']}, indent=2) + '\n')
        print(json.dumps({'status': 'complete', 'splits': report['splits'], 'seconds': report['duration_seconds'],
                          'output': str(args.output)}, indent=2), flush=True)
    except Exception as exc:
        status_path.write_text(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
