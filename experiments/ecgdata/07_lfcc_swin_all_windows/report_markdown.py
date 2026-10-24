"""Write reports/report.md from reports/analysis.json (reads saved numbers only).

Replaces study.write_markdown, which slices 'patient_vote_' keys one character
too far and raises KeyError. study.py is kept byte-identical because its hash
guards the extraction/training cache.
"""
from pathlib import Path
import json

REPORTS = Path(__file__).resolve().parent/'reports'


def pct(x):
    return f'{100*x:.2f}%'


def main():
    a = json.loads((REPORTS/'analysis.json').read_text())
    lines = ['# ECG-07 all-window LFCC/Swin — results', '',
             'Generated from saved out-of-fold predictions. Swin rows are means over seeds 0/1/2; '
             'logistic is deterministic. Intervals: 2,000 paired patient-bootstrap draws (seed 20260925), pointwise, '
             'conditional on saved predictions, without retraining or multiplicity adjustment.', '']
    for set_name, title in [('v1_windows', 'Matched evaluation on the 1,620 v1 windows (primary)'),
                            ('all_windows', 'All 21,222 held-out windows')]:
        r = a['results'][set_name]
        lines += [f'## {title}', '',
                  '| Arm | Window accuracy | Window macro-F1 | Patient-vote accuracy | Patient-vote macro-F1 | Window recall ARR / CHF / NSR |',
                  '|---|---:|---:|---:|---:|---|']
        for arm, item in r['arms'].items():
            w, v = item['mean']['window'], item['mean']['patient_vote']
            lines.append(f"| {arm} | {pct(w['accuracy'])} | {w['macro_f1']:.4f} | {pct(v['accuracy'])} | {v['macro_f1']:.4f} | "
                         + ' / '.join(pct(x) for x in w['sensitivity']) + ' |')
        lines += ['', '| Paired difference | Metric | Observed | 95% interval | Draws > 0 |', '|---|---|---:|---:|---:|']
        for pair, metrics in r['bootstrap']['paired_differences'].items():
            a_name, b_name = pair.split('_minus_')
            for key, d in metrics.items():
                for prefix, scope in (('window_', 'window'), ('patient_vote_', 'patient_vote')):
                    if key.startswith(prefix):
                        metric = key[len(prefix):]
                        break
                observed = r['arms'][a_name]['mean'][scope][metric]-r['arms'][b_name]['mean'][scope][metric]
                fmt = (lambda x: f'{100*x:+.2f} pp') if metric == 'accuracy' else (lambda x: f'{x:+.4f}')
                lo, hi = d['interval_95']
                lines.append(f"| {a_name} − {b_name} | {key} | {fmt(observed)} | [{fmt(lo)}, {fmt(hi)}] | {100*d['fraction_positive']:.1f}% |")
        lines += ['', f"Valid bootstrap draws: {r['bootstrap']['valid_draws']}/{r['bootstrap']['repetitions']}.", '']
    total = sum(f['seconds'] for f in a['fits'])
    lines += ['## Fits', '', f"20 fits; summed fit time {total:.1f} s. Per-fit rows: [fits.csv](fits.csv). "
              'Per-seed metrics: [metrics.csv](metrics.csv). Full analysis: [analysis.json](analysis.json).', '']
    (REPORTS/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
