"""Bounded, resumable 64k assessment; completion does not imply gate success."""
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import time
import traceback

from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc.scripts.eyeball_pilot import one, perturbations
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset, LEADS
from ecgvmd.rotational import RotationalError, analytic_descriptors
import numpy as np

SPEC = C.BASE / 'protocols/eyeball_omi_engineering_retry64k_v1.json'
OUT = C.BASE / 'results/eyeball_omi_pilot_retry64k_v1'
PARENT = C.BASE / 'results/eyeball_omi_pilot_retry16k_v1'
DIAGNOSTIC = C.BASE / 'results/eyeball_stopping_diagnostic_v1'


def input_digest(signal):
    return C.hashlib.sha256(np.ascontiguousarray(signal).tobytes()).hexdigest()


def compatible(old, new):
    """Raising only an unused iteration ceiling preserves a converged result."""
    for key in ('input', 'analytic', 'minimum_seconds'):
        if old[key] != new[key]:
            raise ValueError(f'Changed numerical setting: {key}')
    a, b = copy.deepcopy(old['emd']), copy.deepcopy(new['emd'])
    old_cap = a['parameters'].pop('MAX_ITERATION')
    new_cap = b['parameters'].pop('MAX_ITERATION')
    a.pop('failure', None); b.pop('failure', None)
    if a != b or old_cap > new_cap:
        raise ValueError('Reuse requires unchanged EMD and a nondecreasing ceiling')


def check_manifest(folder):
    value = C.read(folder / 'manifest.json')
    digest = C.sha(folder / 'manifest.json')
    checksum = folder / 'manifest.sha256'
    if checksum.exists() and checksum.read_text().strip() != digest:
        raise ValueError('Changed parent manifest')
    if value['environment'] != C.environment():
        raise ValueError('Changed parent numerical environment')
    for name, expected in value['code_sha256'].items():
        if C.sha(C.ROOT / name) != expected:
            raise ValueError(f'Changed parent source: {name}')
    return value, digest


def context(out):
    spec = C.read(SPEC)
    parent, parent_sha = check_manifest(PARENT)
    diagnostic, diagnostic_sha = check_manifest(DIAGNOSTIC)
    compatible(parent['protocol'], spec)
    if not C.verified(DIAGNOSTIC, diagnostic_sha):
        raise ValueError('Diagnostic is not complete')
    paths = [Path(__file__), SPEC, C.BASE / 'tests/test_eyeball_assessment.py']
    value = dict(protocol=spec, environment=C.environment(),
        code_sha256={**C.source_hashes(), **{str(p.relative_to(C.ROOT)): C.sha(p) for p in paths}},
        parent_manifest_sha256=parent_sha, diagnostic_manifest_sha256=diagnostic_sha,
        diagnostic_completion_sha256=C.sha(DIAGNOSTIC / 'completed.json'),
        source_manifest_sha256=C.sha(C.SOURCE / 'manifest.json'),
        splits_sha256=C.sha(C.SOURCE / 'splits.csv'))
    path = out / 'manifest.json'
    if path.exists():
        if C.read(path) != value or C.sha(path) != (out / 'manifest.sha256').read_text().strip():
            raise ValueError('Changed assessment context; use a new version')
    else:
        C.write(path, value)
        C.put(out / 'manifest.sha256', (C.sha(path) + '\n').encode())
        for name in value['code_sha256']:
            C.put(out / 'source' / name, (C.ROOT / name).read_bytes())
        for name in ('splits.csv', 'source_records.csv'):
            C.put(out / name, (C.SOURCE / name).read_bytes())
    return spec, C.sha(path), parent, diagnostic


def import_baseline(out, row, signal, spec, digest, parent, diagnostic):
    rid = row['record_id']
    dest = out / 'records' / rid / 'I'
    if C.verified(dest, digest):
        return
    source = PARENT / 'records' / rid / 'I'
    if C.verified(source, C.sha(PARENT / 'manifest.json')):
        compatible(parent['protocol'], spec)
        details = C.read(source / 'details.json')
        if details['input_sha256'] != input_digest(signal):
            raise ValueError('Imported signal hash changed')
        data_path = source / 'features.npz'
    elif rid == diagnostic['record']['record_id']:
        source = DIAGNOSTIC / 'default_64000'
        if row != diagnostic['record'] or not C.verified(source, C.sha(DIAGNOSTIC / 'manifest.json')):
            raise ValueError('Diagnostic input or completion changed')
        details = C.read(source / 'details.json')
        if not details['numerically_valid'] or not details['uninstrumented_exact_match']:
            raise ValueError('Cannot reuse invalid diagnostic')
        old_spec = copy.deepcopy(diagnostic['base_spec'])
        old_spec['emd']['parameters'].update({k: v for k, v in details['configuration'].items() if k != 'name'})
        compatible(old_spec, spec)
        data_path = source / 'decomposition.npz'
        details.update(total_seconds=details['seconds'], timing_scope='original diagnostic decomposition only',
            input_sha256=input_digest(signal), samples=len(signal),
            identity=dict(record_id=rid, patient_id=row['patient_id'], lead='I', partition='fit'))
    else:
        return
    if details['components'] != 4 or any(s['capped'] for s in details['sifting']):
        raise ValueError('Cannot import capped or incomplete decomposition')
    data = C.arrays(data_path)
    np.testing.assert_allclose(data['modes'].sum(0) + data['residual'], signal, atol=1e-12, rtol=1e-12)
    values, _, trajectory = analytic_descriptors(data['modes'], spec['input']['fs'], spec['analytic'])
    np.testing.assert_array_equal(values, data['features'])
    np.testing.assert_array_equal(trajectory, data['trajectory'])
    C.put(dest / 'features.npz', data_path.read_bytes())
    C.write(dest / 'details.json', details)
    C.write(dest / 'reuse.json', dict(source=str(source.relative_to(C.ROOT)),
        source_completion_sha256=C.sha(source / 'completed.json'),
        source_data_sha256=C.sha(data_path), time_utc=C.now(), rationale='Verified uncapped identical algorithm'))
    C.complete(dest, ['features.npz', 'details.json', 'reuse.json'], digest)
    C.event(dest, 'imported', source=str(source.relative_to(C.ROOT)))


def task(folder, signal, spec, digest, identity, retain=False):
    """A numerical failure is terminal under this frozen context, and is reusable."""
    folder = Path(folder)
    signature = dict(context_sha=digest, input_sha256=input_digest(signal), identity=identity, retain=retain)
    receipt = folder / 'terminal.json'
    if receipt.exists():
        saved = C.read(receipt)
        if saved['signature'] != signature:
            raise ValueError('Changed task identity')
        for name, expected in saved['files'].items():
            if C.sha(folder / name) != expected:
                raise ValueError('Changed task evidence')
        if saved['outcome'] == 'success' and not C.verified(folder, digest):
            raise ValueError('Missing successful task completion')
        return saved
    started = time.perf_counter()
    try:
        _, details = one(folder, signal, spec, digest, identity, retain=retain)
        if details['input_sha256'] != signature['input_sha256']:
            raise ValueError('Saved result belongs to another signal')
        saved = dict(signature=signature, outcome='success', files={'completed.json': C.sha(folder / 'completed.json')},
            seconds_this_attempt=time.perf_counter()-started, extraction_seconds=details['total_seconds'])
    except RotationalError as exc:
        failures = sorted(folder.glob('failure_*.json'))
        saved = dict(signature=signature, outcome='numerical_failure', error=str(exc),
            diagnostics=exc.diagnostics, seconds_this_attempt=time.perf_counter()-started,
            files={p.name: C.sha(p) for p in failures})
    C.write(receipt, saved)
    return saved


def execute(pool, tasks, out, stage):
    pending = {pool.submit(task, *args): args[4] for args in tasks}
    outcomes = []
    for future in as_completed(pending):
        result = future.result()
        outcomes.append(result)
        failed = sum(r['outcome'] != 'success' for r in outcomes)
        C.write(out / 'status.json', dict(status='running', stage=stage,
            finished=len(outcomes), total=len(tasks), numerical_failures=failed, time_utc=C.now()))
        if len(outcomes) % 16 == 0 or result['outcome'] != 'success' or len(outcomes) == len(tasks):
            print(f'{stage}: {len(outcomes)}/{len(tasks)}; numerical failures={failed}', flush=True)
    return sorted(outcomes, key=lambda r: C.json.dumps(r['signature']['identity'], sort_keys=True))


def gallery(out, rows, digest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    saved = {}
    for row in rows:
        folder = out / 'records' / row['record_id'] / 'I'
        if C.verified(folder, digest):
            saved[row['record_id']] = C.arrays(folder / 'features.npz')
    extent = max((float(np.max(np.abs(np.r_[a['trajectory'].real, a['trajectory'].imag])))
                  for a in saved.values()), default=1.)
    fig, axes = plt.subplots(4, 8, figsize=(20, 10), sharex=True, sharey=True)
    for ax, row in zip(axes.flat, rows):
        rid = row['record_id']
        if rid in saved:
            data = saved[rid]; z = data['trajectory']
            ax.plot(z.real, z.imag, linewidth=.3, alpha=.65)
            ax.plot(*data['features'][10:], 'rx', markersize=4)
        else:
            ax.text(.5, .5, 'Extraction failed', transform=ax.transAxes, ha='center', fontsize=7)
        ax.set(title=rid, xlim=(-extent*1.03, extent*1.03), ylim=(-extent*1.03, extent*1.03), aspect='equal')
    fig.suptitle('Lead I: all 32 fixed fit patients; common axes; red = arc-length centroid')
    fig.supxlabel('Real (mV)'); fig.supylabel('Imaginary (mV)'); fig.tight_layout()
    fig.savefig(out / 'pilot_geometry.png', dpi=140); plt.close(fig)


def run(out=OUT):
    started = time.perf_counter()
    with C.lock(out):
        spec, digest, parent, diagnostic = context(out)
        if C.verified(out, digest):
            print('Verified completed assessment; no extraction repeated', flush=True)
            return
        rows = C.selection(C.split_rows(), spec['pilot']['patients'], spec['pilot']['selection_id'])
        with (PARENT / 'selection.csv').open(newline='') as stream:
            if list(C.csv.DictReader(stream)) != rows:
                raise ValueError('Original sample changed')
        C.csv_write(out / 'selection.csv', rows)
        C.event(out, 'started', context_sha=digest)
        C.write(out / 'status.json', dict(status='running', stage='verify_inputs', time_utc=C.now()))
        try:
            signals = {}
            with ACSDataset(target='OMI') as ds:
                for row in rows:
                    r = ds.load_record(row['record_id'])
                    if (r.info.patient_id != row['patient_id'] or r.info.split != 'train'
                            or r.decision.waveform_sha256 != row['waveform_sha256']):
                        raise ValueError('Fit input changed')
                    signals[row['record_id']] = r.signal_mV.T.copy()
            for row in rows:
                import_baseline(out, row, signals[row['record_id']][0], spec, digest, parent, diagnostic)
            with ProcessPoolExecutor(max_workers=spec['execution']['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
                tasks = []
                for row in rows:
                    rid = row['record_id']
                    for j, lead in enumerate(LEADS):
                        identity = dict(record_id=rid, patient_id=row['patient_id'], lead=lead, partition='fit')
                        tasks.append((out/'records'/rid/lead, signals[rid][j], spec, digest, identity, lead=='I'))
                baseline = execute(pool, tasks, out, 'baseline')
                C.write(out / 'baseline.json', baseline)
                tasks = []
                for row in rows:
                    rid = row['record_id']
                    if C.verified(out/'records'/rid/'I', digest):
                        tasks.append((out/'repeats'/rid, signals[rid][0], spec, digest,
                            dict(record_id=rid, patient_id=row['patient_id'], lead='I', repeat=True), True))
                repeats = execute(pool, tasks, out, 'exact_repeats')
                for result in repeats:
                    if result['outcome'] != 'success':
                        raise ValueError('Previously converged signal failed on repeat')
                    rid = result['signature']['identity']['record_id']
                    a = C.arrays(out/'records'/rid/'I'/'features.npz')
                    b = C.arrays(out/'repeats'/rid/'features.npz')
                    for key in ('features', 'modes', 'residual', 'trajectory'):
                        np.testing.assert_array_equal(a[key], b[key])
                    result['exact_match'] = True
                C.write(out/'repeats.json', repeats)
                tasks = []
                for i, row in enumerate(rows):
                    rid = row['record_id']; seed = spec['pilot']['perturbation_seed'] + i
                    for name, x in perturbations(signals[rid][0], 500, seed).items():
                        tasks.append((out/'perturbations'/rid/name, x, spec, digest,
                            dict(record_id=rid, patient_id=row['patient_id'], lead='I', perturbation=name, seed=seed), False))
                perturbed = execute(pool, tasks, out, 'perturbations')
            comparisons = []
            for result in perturbed:
                identity = result['signature']['identity']; rid = identity['record_id']; name = identity['perturbation']
                item = dict(identity=identity, outcome=result['outcome'], comparison_available=False)
                if result['outcome'] == 'success' and C.verified(out/'records'/rid/'I', digest):
                    a = C.arrays(out/'records'/rid/'I'/'features.npz')['features']
                    b = C.arrays(out/'perturbations'/rid/name/'features.npz')['features']
                    delta = 2*np.abs(a[:10]-b[:10])/(np.abs(a[:10])+np.abs(b[:10])+1e-12)
                    shift = float(np.linalg.norm(a[10:]-b[10:])/max(a[9], 1e-12))
                    item.update(comparison_available=True, symmetric_relative_changes=delta.tolist(),
                        centroid_shift_over_envelope=shift, feature_warnings=int((delta>spec['pilot']['relative_feature_warning']).sum()),
                        centroid_warning=shift>.1)
                comparisons.append(item)
            C.write(out/'perturbations.json', perturbed)
            C.write(out/'comparisons.json', comparisons)
            success = [r for r in baseline if r['outcome']=='success']
            failed = [r for r in baseline if r['outcome']!='success']
            details = [C.read(out/'records'/r['signature']['identity']['record_id']/r['signature']['identity']['lead']/'details.json') for r in success]
            all_times = [r.get('extraction_seconds', r['seconds_this_attempt']) for r in baseline]
            gate = not failed and len(repeats)==len(rows)
            summary = dict(status='complete', assessment_complete=True, baseline_gate_passed=gate,
                full_run_authorized_by_engineering=False,
                decision='Requires documented perturbation review before full run' if gate else 'Do not scale or train: baseline engineering gate failed',
                baseline_patients=len(rows), baseline_attempts=len(baseline), baseline_successes=len(success), baseline_failures=len(failed),
                lead_I_successes=sum(r['signature']['identity']['lead']=='I' for r in success),
                exact_repeat_checks=len(repeats), reused_baselines=len(list((out/'records').rglob('reuse.json'))),
                perturbation_attempts=len(perturbed), perturbation_successes=sum(r['outcome']=='success' for r in perturbed),
                perturbation_failures=sum(r['outcome']!='success' for r in perturbed),
                available_perturbation_comparisons=sum(c['comparison_available'] for c in comparisons),
                perturbation_feature_warnings=sum(c.get('feature_warnings',0) for c in comparisons),
                perturbation_centroid_warnings=sum(c.get('centroid_warning',False) for c in comparisons),
                maximum_converged_sifts=max((s['sifts'] for d in details for s in d['sifting']),default=None),
                maximum_reconstruction_error_mV=max((d['reconstruction_max_abs_mV'] for d in details),default=None),
                summed_baseline_task_seconds=sum(all_times),
                baseline_task_seconds_quantiles=dict(zip(['min','median','p90','max'],np.quantile(all_times,[0,.5,.9,1]).tolist())),
                runtime_caveat='Four simultaneous workers; imported durations measured earlier, diagnostic import decomposition-only; not serial throughput',
                seconds_this_session=time.perf_counter()-started, models_trained=0, official_test_processed=False)
            gallery(out, rows, digest)
            C.write(out/'summary.json', summary)
            files = [str(p.relative_to(out)) for p in sorted(out.rglob('*')) if p.is_file()
                     and p.relative_to(out).as_posix() not in ('completed.json','status.json','attempts.jsonl','.writer.lock')]
            C.complete(out, files, digest)
            C.write(out/'status.json', summary)
            C.event(out,'complete', baseline_gate_passed=gate, baseline_failures=len(failed))
            print(C.json.dumps(summary, indent=2), flush=True)
        except BaseException:
            C.write(out/'status.json', dict(status='failed', time_utc=C.now(), traceback=traceback.format_exc()))
            C.event(out,'failed', traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    run()
