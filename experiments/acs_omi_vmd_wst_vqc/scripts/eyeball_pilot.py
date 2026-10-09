"""Run/resume the declared fit-patient rotational morphology engineering pilot."""
import argparse
import resource
import time
import traceback
import warnings

from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset, LEADS
from ecgvmd.rotational import extract, RotationalError, NAMES
import numpy as np


def one(folder, signal, spec, context, identity, *, retain=False):
    folder.mkdir(parents=True, exist_ok=True)
    if C.verified(folder, context):
        return C.arrays(folder / 'features.npz')['features'], C.read(folder / 'details.json')
    started = time.perf_counter()
    C.event(folder, 'started', **identity)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            values, details, saved = extract(signal, spec['input']['fs'], spec, retain=retain)
        details.update(identity=identity, warnings=[str(w.message) for w in caught],
            input_sha256=C.hashlib.sha256(np.ascontiguousarray(signal).tobytes()).hexdigest(),
            samples=len(signal), process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        C.npz(folder / 'features.npz', features=values, names=np.array(NAMES), **(saved or {}))
        C.write(folder / 'details.json', details)
        C.complete(folder, ['features.npz', 'details.json'], context)
        C.event(folder, 'complete', seconds=time.perf_counter() - started, **identity)
        return values, details
    except BaseException as exc:
        record = dict(identity=identity, traceback=traceback.format_exc(),
                      diagnostics=getattr(exc, 'diagnostics', {}), seconds=time.perf_counter() - started)
        C.write(folder / f'failure_{time.time_ns()}.json', record)
        C.event(folder, 'failed', **record)
        raise


def perturbations(x, fs, seed):
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(x ** 2))
    return dict(first_half=x[:len(x)//2], last_half=x[len(x)//2:],
        noise_30dB=x + rng.normal(0, rms * 10**(-30/20), len(x)),
        noise_20dB=x + rng.normal(0, rms * 10**(-20/20), len(x)),
        amplitude_x2=2*x, polarity=-x, crop_100ms=x[round(.1*fs):-round(.1*fs)])


def make_plots(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = [C.arrays(out / 'records' / r['record_id'] / 'I' / 'features.npz') for r in rows]
    extent = max(float(np.max(np.abs(np.r_[d['trajectory'].real, d['trajectory'].imag]))) for d in values)
    fig, axes = plt.subplots(4, 8, figsize=(20, 10), sharex=True, sharey=True)
    for ax, row, data in zip(axes.flat, rows, values):
        z = data['trajectory']
        ax.plot(z.real, z.imag, linewidth=.3, alpha=.65)
        ax.plot(data['features'][10], data['features'][11], 'rx', markersize=4)
        ax.set_title(row['record_id'], fontsize=8)
        ax.set(xlim=(-extent*1.03, extent*1.03), ylim=(-extent*1.03, extent*1.03), aspect='equal')
    fig.suptitle('Lead I: fixed fit-patient sample; identical axes in mV; red = arc-length centroid')
    fig.supxlabel('Real (mV)'); fig.supylabel('Imaginary (mV)')
    fig.tight_layout()
    fig.savefig(out / 'pilot_geometry.png', dpi=140)
    plt.close(fig)


def run(out):
    started = time.perf_counter()
    with C.lock(out):
        spec, digest, all_rows = C.context(out, create=True)
        if C.verified(out, digest):
            print('Verified completed pilot; no extraction repeated', flush=True)
            return
        rows = C.selection(all_rows, spec['pilot']['patients'], spec['id'])
        if (out / 'selection.csv').exists():
            with (out / 'selection.csv').open(newline='') as stream:
                if list(C.csv.DictReader(stream)) != rows:
                    raise ValueError('Pilot selection changed')
        else:
            C.csv_write(out / 'selection.csv', rows)
        C.event(out, 'started', selected_patients=len(rows), context_sha=digest)
        C.write(out / 'status.json', dict(status='running', stage='baseline_lead_I', time_utc=C.now()))
        try:
            signals = {}
            with ACSDataset(target='OMI') as ds:
                for row in rows:
                    record = ds.load_record(row['record_id'])
                    if (record.info.patient_id != row['patient_id'] or record.info.split != 'train'
                            or record.decision.waveform_sha256 != row['waveform_sha256']):
                        raise ValueError('Pilot waveform identity changed')
                    signals[row['record_id']] = record.signal_mV.T.copy()
                for stage, leads in [('baseline_lead_I', ['I']), ('baseline_other_leads', LEADS[1:])]:
                    C.write(out / 'status.json', dict(status='running', stage=stage, time_utc=C.now()))
                    for i, row in enumerate(rows):
                        rid = row['record_id']
                        for lead in leads:
                            one(out / 'records' / rid / lead, signals[rid][LEADS.index(lead)], spec, digest,
                                dict(record_id=rid, patient_id=row['patient_id'], lead=lead, partition='fit'), retain=lead=='I')
                        print(f'{stage}: {i+1}/{len(rows)} ECGs', flush=True)
            # Repeat every Lead-I input and compare numerical outputs, not timings.
            for row in rows:
                rid = row['record_id']
                saved = C.arrays(out / 'records' / rid / 'I' / 'features.npz')
                repeated, _, retained = extract(signals[rid][0], 500, spec, retain=True)
                np.testing.assert_array_equal(repeated, saved['features'])
                np.testing.assert_array_equal(retained['modes'], saved['modes'])
            C.write(out / 'status.json', dict(status='running', stage='perturbations', time_utc=C.now()))
            comparisons = []
            for i, row in enumerate(rows):
                rid = row['record_id']
                baseline = C.arrays(out / 'records' / rid / 'I' / 'features.npz')['features']
                for name, x in perturbations(signals[rid][0], 500, spec['pilot']['perturbation_seed'] + i).items():
                    try:
                        features, _ = one(out / 'perturbations' / rid / name, x, spec, digest,
                            dict(record_id=rid, patient_id=row['patient_id'], lead='I', perturbation=name, seed=spec['pilot']['perturbation_seed']+i))
                        delta = 2 * np.abs(features[:10] - baseline[:10]) / (np.abs(features[:10]) + np.abs(baseline[:10]) + 1e-12)
                        centroid_shift = float(np.linalg.norm(features[10:] - baseline[10:]) / max(baseline[9], 1e-12))
                        comparisons.append(dict(record_id=rid, perturbation=name, status='complete',
                            symmetric_relative_changes=delta.tolist(), centroid_shift_over_envelope=centroid_shift,
                            warnings=int((delta > spec['pilot']['relative_feature_warning']).sum()), centroid_warning=centroid_shift>.1))
                    except RotationalError as exc:
                        comparisons.append(dict(record_id=rid, perturbation=name, status='failed', error=str(exc)))
                print(f'Perturbations: {i+1}/{len(rows)} ECGs', flush=True)
            details = [C.read(out / 'records' / r['record_id'] / lead / 'details.json') for r in rows for lead in LEADS]
            elapsed = sum(d['total_seconds'] for d in details)
            C.write(out / 'perturbations.json', comparisons)
            summary = dict(status='complete', baseline_ECGs=len(rows), baseline_leads=len(details),
                exact_repeat_checks=len(rows), perturbation_attempts=len(comparisons),
                perturbation_failures=sum(c['status']=='failed' for c in comparisons),
                perturbation_feature_warnings=sum(c.get('warnings',0) for c in comparisons),
                perturbation_centroid_warnings=sum(c.get('centroid_warning',False) for c in comparisons),
                summed_baseline_extraction_seconds=elapsed, mean_seconds_per_ECG=elapsed/len(rows),
                max_sifts=max(s['sifts'] for d in details for s in d['sifting']),
                maximum_reconstruction_error_mV=max(d['reconstruction_max_abs_mV'] for d in details),
                seconds_this_session=time.perf_counter()-started, models_trained=0, official_test_processed=False,
                interpretation='Engineering checks only; perturbation sensitivity does not establish predictive value')
            C.write(out / 'summary.json', summary)
            make_plots(out, rows)
            names=['selection.csv', 'summary.json', 'perturbations.json', 'pilot_geometry.png', 'splits.csv', 'source_records.csv']
            names += [str(p.relative_to(out)) for directory in ('records','perturbations')
                      for p in sorted((out / directory).rglob('*')) if p.is_file()]
            C.complete(out, names, digest)
            C.write(out / 'status.json', summary)
            C.event(out, 'complete', **{k:v for k,v in summary.items() if k!='status'})
            print(C.json.dumps(summary, indent=2), flush=True)
        except BaseException:
            failure=dict(status='failed', traceback=traceback.format_exc(), time_utc=C.now())
            C.write(out / 'status.json', failure)
            C.event(out, 'failed', traceback=failure['traceback'])
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=C.Path, default=C.BASE / 'results/eyeball_omi_pilot_v1')
    args = parser.parse_args()
    run(args.out)
