"""Tie-preserving sensitivity targets and paired patient-cluster intervals."""
import numpy as np

FIELDS = ('sensitivity', 'specificity', 'precision', 'accuracy', 'tp', 'fn', 'tn', 'fp',
          'false_alarms_per_1000_negative', 'positive_fraction')


def validate(y, score, weights=None):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if (y.ndim != 1 or score.shape != y.shape or weights.shape != y.shape
            or not set(np.unique(y)) <= {0, 1} or not np.isfinite(score).all()
            or np.any((score < 0) | (score > 1)) or not np.isfinite(weights).all()
            or np.any(weights < 0) or any(weights[y == k].sum() <= 0 for k in (0, 1))):
        raise ValueError('Need finite probabilities and positive weight for both classes')
    return y.astype(int), score, weights


def counts(tp, fp, positives, negatives):
    fn, tn = positives - tp, negatives - fp
    return dict(zip(FIELDS, (tp / positives, tn / negatives,
                tp / (tp + fp) if tp + fp else 0., (tp + tn) / (positives + negatives),
                tp, fn, tn, fp, 1000 * fp / negatives, (tp + fp) / (positives + negatives))))


def at_threshold(y, score, threshold, weights=None):
    y, score, weights = validate(y, score, weights)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid threshold')
    pred = score >= threshold
    return counts(float(weights[pred & (y == 1)].sum()), float(weights[pred & (y == 0)].sum()),
                  float(weights[y == 1].sum()), float(weights[y == 0].sum()))


class Curve:
    def __init__(self, y, score):
        self.y, self.score, _ = validate(y, score)
        self.order = np.argsort(-self.score, kind='stable')
        ordered = self.score[self.order]
        self.ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(ordered) - 1]
        self.thresholds = ordered[self.ends]

    def operating_points(self, targets, weights=None):
        if any(not isinstance(t, (int, np.integer)) or not 0 < t <= 100 for t in targets):
            raise ValueError('Targets must be integer percentages in (0, 100]')
        y, _, weights = validate(self.y, self.score, weights)
        tp = np.cumsum((weights * y)[self.order])[self.ends]
        fp = np.cumsum((weights * (1 - y))[self.order])[self.ends]
        # Integer percentage comparison avoids ceil/floating-point boundary surprises.
        indices = np.searchsorted(tp * 100, np.asarray(targets) * tp[-1], side='left')
        return [dict(threshold=float(self.thresholds[i]),
                     **counts(float(tp[i]), float(fp[i]), float(tp[-1]), float(fp[-1])))
                for i in indices]


def groups_for(scores):
    groups = {}
    for name in scores:
        group = name.rsplit('_seed', 1)[0] if '_seed' in name else name
        groups.setdefault(group, []).append(name)
    for group, members in groups.items():
        if group.endswith(('/vqc', '/swin')) and members != [group + f'_seed{s}' for s in (0, 1, 2)]:
            raise ValueError(f'Missing or reordered initialization seeds: {group}')
    return groups


def analyze(y, patients, scores, targets, draws, seed, thresholds=None):
    """Fixed thresholds for new fits; reselect per draw only for descriptive curves."""
    if len(patients) != len(y) or not scores or draws < 1:
        raise ValueError('Invalid bootstrap inputs')
    groups = groups_for(scores)
    curves = {name: Curve(y, s) for name, s in scores.items()}

    def evaluate(weights=None):
        return {name: (curve.operating_points(targets, weights) if thresholds is None else
                      [dict(threshold=float(t), **at_threshold(y, scores[name], t, weights))
                       for t in thresholds[name]]) for name, curve in curves.items()}

    point = evaluate()
    def group_values(values):
        return {name: np.mean([[[row[k] for k in FIELDS] for row in values[m]]
                              for m in members], axis=0) for name, members in groups.items()}
    grouped = group_values(point)
    per_draw = {name: [] for name in scores}
    group_draws = {name: [] for name in groups}
    unique, inverse = np.unique(patients, return_inverse=True)
    rng, skipped = np.random.default_rng(seed), 0
    for i in range(draws):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[inverse]
        if any(weights[np.asarray(y) == k].sum() == 0 for k in (0, 1)):
            skipped += 1
            continue
        values = evaluate(weights)
        for name, rows in values.items():
            per_draw[name].append([[r[k] for k in ('threshold',) + FIELDS] for r in rows])
        for name, values in group_values(values).items():
            group_draws[name].append(values)
        if (i + 1) % 500 == 0:
            print(f'Patient bootstrap {i + 1}/{draws}', flush=True)
    if skipped == draws:
        raise ValueError('All bootstrap samples have only one class')

    def intervals(values, fields):
        lo, hi = np.percentile(values, [2.5, 97.5], axis=0)
        return [{k: [float(a), float(b)] for k, a, b in zip(fields, l, h)} for l, h in zip(lo, hi)]

    pairs = {}
    reference = next((n for n in ('new/lfcc/logistic', 'balanced/lfcc/logistic') if n in groups), None)
    if reference:
        for name in groups:
            if name == reference:
                continue
            pairs[name + ' minus ' + reference] = dict(
                point=[dict(zip(FIELDS, row)) for row in (grouped[name] - grouped[reference]).tolist()],
                intervals=intervals(np.asarray(group_draws[name]) - group_draws[reference], FIELDS))
    result = dict(targets_percent=targets, per_model=point,
        groups={name: [dict(zip(FIELDS, row)) for row in values.tolist()] for name, values in grouped.items()},
        per_model_intervals={n: intervals(v, ('threshold',) + FIELDS) for n, v in per_draw.items()},
        group_intervals={n: intervals(v, FIELDS) for n, v in group_draws.items()}, paired_differences=pairs,
        bootstrap=dict(requested=draws, valid=draws-skipped, single_class_discarded=skipped, seed=seed,
                       unit='patient, all ECGs together', thresholds_fixed=thresholds is not None))
    # Preserve draws so paired estimates can be audited without rerunning sampling.
    return result, {name: np.asarray(values) for name, values in per_draw.items()}
