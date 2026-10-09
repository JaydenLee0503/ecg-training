"""Descriptive engineering report; no model fitting or held-out scoring."""
from pathlib import Path
import numpy as np
from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc.loader import LEADS
from experiments.acs_omi_vmd_wst_vqc.scripts.eyeball_pilot_assessment import OUT, check_manifest

REPORT = C.BASE / 'reports/eyeball_assessment_2026-10-08'


def run():
    manifest, digest = check_manifest(OUT)
    if not C.verified(OUT, digest):
        raise ValueError('Complete the assessment before reporting')
    context = dict(assessment_manifest_sha256=digest,
        assessment_completion_sha256=C.sha(OUT/'completed.json'),
        reporting_source_sha256=C.sha(__file__), environment=C.environment(),
        bootstrap=dict(unit='patient', draws=2000, seed=20261008,
            statistic='fraction of baseline leads with numerical failure', interval='percentile 2.5,97.5'),
        scope='Descriptive numerical feasibility; no classifier prediction or clinical performance')
    with C.lock(REPORT):
        if (REPORT/'manifest.json').exists():
            if C.read(REPORT/'manifest.json') != context:
                raise ValueError('Changed report context')
        else:
            C.write(REPORT/'manifest.json', context)
            C.put(REPORT/'reporting_source.py', Path(__file__).read_bytes())
        report_digest = C.sha(REPORT/'manifest.json')
        if C.verified(REPORT, report_digest):
            print('Verified completed assessment report'); return
        completion = C.read(OUT/'completed.json')
        inventory = {name: dict(sha256=expected, bytes=(OUT/name).stat().st_size)
                     for name, expected in completion['files'].items()}
        C.write(REPORT/'artifact_inventory.json', dict(time_utc=C.now(),
            root=str(OUT.relative_to(C.ROOT)), entries_checked=len(inventory), files=inventory, mismatches=[]))
        summary = C.read(OUT/'summary.json')
        baseline = C.read(OUT/'baseline.json')
        perturbations = C.read(OUT/'perturbations.json')
        comparisons = C.read(OUT/'comparisons.json')
        with (OUT/'selection.csv').open(newline='') as stream:
            cohort = list(C.csv.DictReader(stream))
        expected = {(r['record_id'], lead) for r in cohort for lead in LEADS}
        actual = [(r['signature']['identity']['record_id'], r['signature']['identity']['lead']) for r in baseline]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Missing or duplicate baseline outcomes')
        variants = {'first_half','last_half','noise_30dB','noise_20dB','amplitude_x2','polarity','crop_100ms'}
        expected_perturbations = {(r['record_id'], name) for r in cohort for name in variants}
        actual_perturbations = [(r['signature']['identity']['record_id'], r['signature']['identity']['perturbation']) for r in perturbations]
        if len(actual_perturbations) != len(expected_perturbations) or set(actual_perturbations) != expected_perturbations:
            raise ValueError('Missing or duplicate perturbation outcomes')
        if len({r['patient_id'] for r in cohort}) != 32 or any(r['partition']!='fit' for r in cohort):
            raise ValueError('Changed pilot patients')
        patient_index = {r['record_id']: i for i, r in enumerate(cohort)}
        failures = np.zeros((len(cohort), len(LEADS)))
        flat = []
        for r in baseline:
            identity = r['signature']['identity']; rid = identity['record_id']; lead = identity['lead']
            failed = r['outcome'] != 'success'
            failures[patient_index[rid], LEADS.index(lead)] = failed
            flat.append(dict(**identity, outcome=r['outcome'], error=r.get('error',''),
                extraction_or_failure_seconds=r.get('extraction_seconds',r['seconds_this_attempt'])))
        C.csv_write(REPORT/'baseline_outcomes.csv', flat)
        rng = np.random.default_rng(20261008)
        draws = failures.mean(1)[rng.integers(0,len(cohort),size=(2000,len(cohort)))].mean(1)
        interval = np.quantile(draws,[.025,.975]).tolist()
        C.npz(REPORT/'bootstrap_draws.npz', baseline_lead_failure_fraction=draws)
        if int(failures.sum()) != summary['baseline_failures']:
            raise ValueError('Summary and saved outcomes disagree')
        negative_frequencies = []
        for r in baseline:
            if r['outcome']=='success':
                identity = r['signature']['identity']
                details = C.read(OUT/'records'/identity['record_id']/identity['lead']/'details.json')
                negative_frequencies.append(details['analytic']['negative_frequency_fraction'])
        groups = {}
        for c in comparisons:
            name = c['identity']['perturbation']
            group = groups.setdefault(name, dict(attempts=0, successes=0, comparable=0,
                any_feature_warning=0, centroid_warning=0, adjusted_max_relative_changes=[]))
            group['attempts'] += 1
            group['successes'] += c['outcome']=='success'
            group['comparable'] += c['comparison_available']
            group['any_feature_warning'] += c.get('feature_warnings',0)>0
            group['centroid_warning'] += c.get('centroid_warning',False)
            if name in ('amplitude_x2','polarity') and c['comparison_available']:
                rid = c['identity']['record_id']
                a = C.arrays(OUT/'records'/rid/'I'/'features.npz')['features']
                b = C.arrays(OUT/'perturbations'/rid/name/'features.npz')['features'].copy()
                # Compare after undoing the expected physical coordinate change.
                if name=='amplitude_x2': b[[4,5,6,7,9,10,11]] /= 2
                else: b[10:] *= -1
                delta = 2*np.abs(a[:10]-b[:10])/(np.abs(a[:10])+np.abs(b[:10])+1e-12)
                shift = np.linalg.norm(a[10:]-b[10:])/max(a[9],1e-12)
                group['adjusted_max_relative_changes'].append(dict(record_id=rid,
                    max_first_ten=float(delta.max()), centroid_shift_over_envelope=float(shift)))
        result = dict(summary=summary, cohort=dict(fit_patients=len(cohort), validation_patients=0,
            positive_ECGs=sum(int(r['label']) for r in cohort), sampling='Fixed patient/record hash ranking without outcome use'),
            patients_with_any_baseline_failure=int((failures.sum(1)>0).sum()),
            baseline_failure_fraction=float(failures.mean()), descriptive_patient_bootstrap_interval=interval,
            failures_by_lead=dict(zip(LEADS, failures.sum(0).astype(int).tolist())),
            median_negative_instantaneous_frequency_fractions=np.median(negative_frequencies,axis=0).tolist(),
            perturbations=groups,
            failure_reasons={error: sum(r.get('error')==error for r in baseline+perturbations)
                for error in sorted({r['error'] for r in baseline+perturbations if 'error' in r})})
        C.write(REPORT/'analysis.json', result)
        lines = ['# Eyeball / rotational morphology engineering assessment', '',
            f"**Assessment complete; baseline gate {'passed' if summary['baseline_gate_passed'] else 'failed'}.**",
            f"{summary['baseline_successes']}/{summary['baseline_attempts']} baseline leads succeeded across the unchanged 32 fit patients. "
            f"{summary['baseline_failures']} failed; {result['patients_with_any_baseline_failure']}/32 patients had at least one failed lead.", '',
            f"Lead I succeeded for {summary['lead_I_successes']}/32 patients. "
            f"All {summary['exact_repeat_checks']} attempted repeat checks exactly matched saved features, modes, residual and trajectory. "
            f"{summary['reused_baselines']} baseline outputs were imported with verified provenance.", '',
            f"{summary['perturbation_successes']}/{summary['perturbation_attempts']} perturbations succeeded; "
            f"{summary['available_perturbation_comparisons']} could be compared with successful baselines.", '',
            '**Decision:** '+summary['decision']+'. No Eyeball classifier was fitted; there is no new accuracy estimate.', '',
            '## Protocol and saved evidence', '',
            'Independent 10-second, 500 Hz, physical-mV adaptation of the proposed EMD/Hilbert representation. '
            'Four IMFs, unchanged PyEMD 1.6.4 stopping criteria, 64,000 iteration ceiling, 0.25-second endpoint trim, '
            'signed frequency, and open-trajectory arc-length centroid. This is not a verified reproduction of the preprint. '
            'The cap was amended using fit-only convergence diagnostics, with prior failures retained.', '',
            f"The fixed sample contains {result['cohort']['positive_ECGs']} OMI-positive ECGs; labels did not select patients or numerical settings. "
            'No validation or official-test ECG was processed. Perturbation seeds are 20261005 through 20261036 in the saved cohort order. '
            'No model initialization seeds or predictions exist because fitting did not run.', '',
            'The raw run saves every task outcome, input hash, convergence trace, runtime, source snapshot and attempt ledger. '
            'Completed outputs include four-component reconstruction checks; capped results are not accepted as features. '
            'Missing perturbation comparisons remain missing and are not silently counted as passes.', '',
            '## Convergence', '', '| Lead | Failed / 32 |', '|---|---:|']
        lines += [f'| {lead} | {count} |' for lead,count in result['failures_by_lead'].items()]
        lines += ['', f"Maximum converged baseline sifts: {summary['maximum_converged_sifts']}; "
            f"maximum reconstruction error: {summary['maximum_reconstruction_error_mV']} mV.", '',
            f"Baseline lead failure fraction: {failures.mean():.2%}; descriptive 95% patient-bootstrap interval "
            f"[{interval[0]:.2%}, {interval[1]:.2%}] (2,000 draws, seed 20261008). "
            'This small, repeatedly examined engineering sample does not establish a population failure rate or clinical validity.', '',
            '## Perturbations', '',
            '| Perturbation | Succeeded / 32 | Comparable | Any descriptor warning | Centroid warning |',
            '|---|---:|---:|---:|---:|']
        lines += [f"| {name} | {g['successes']} | {g['comparable']} | {g['any_feature_warning']} | {g['centroid_warning']} |" for name,g in groups.items()]
        lines += ['', 'Warnings use the predeclared raw-feature thresholds: symmetric relative change >0.25 in any of the first ten descriptors, '
            'or centroid displacement / baseline envelope >0.1. Warnings are descriptive, never exclusion criteria. '
            '**Amplitude doubling is expected to change envelope and coordinate features; polarity inversion is expected to negate coordinates.** '
            'Their raw warnings alone do not establish instability. The JSON includes comparisons after undoing those expected transformations. '
            'Shortening, noise and endpoint cropping also change the input; these checks do not measure diagnostic accuracy.', '',
            '## Runtime and limits', '',
            f"Assessment session: {summary['seconds_this_session']:.1f} seconds before gallery/final manifest writing. "
            f"Summed baseline task durations: {summary['summed_baseline_task_seconds']:.1f} seconds. "
            'Four workers ran concurrently; imported timings were measured earlier and the diagnostic import timing covers decomposition only. '
            'These durations are not a controlled serial benchmark and should not be extrapolated as a full-dataset runtime.', '',
            'The numerical and checkpoint tests do not prove the absence of bugs. The result applies to this declared adaptation and iteration bound; '
            'it does not reject rotational morphology generally, establish transform superiority, or establish quantum advantage. '
            'A changed stopping algorithm, preprocessing scheme or feature definition would require a separate documented experiment.', '',
            '[All baseline outcomes](baseline_outcomes.csv) · [Full numerical analysis](analysis.json) · '
            '[Fixed-sample trajectory gallery](pilot_geometry.png) · [Continuation record](../eyeball_continuation_2026-10-08.md)', '']
        C.put(REPORT/'report.md', '\n'.join(lines).encode())
        C.put(REPORT/'pilot_geometry.png', (OUT/'pilot_geometry.png').read_bytes())
        C.complete(REPORT,['manifest.json','reporting_source.py','artifact_inventory.json','analysis.json','bootstrap_draws.npz','baseline_outcomes.csv','report.md','pilot_geometry.png'],report_digest)
        print(C.json.dumps({k:v for k,v in result.items() if k!='perturbations'},indent=2))


if __name__=='__main__':
    run()
