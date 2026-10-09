"""Frozen higher-cap engineering retry on the unchanged 32-patient pilot sample."""
import argparse
import time
import traceback
from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc.scripts.eyeball_pilot import one, perturbations, make_plots
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset, LEADS
from ecgvmd.rotational import extract, RotationalError
import numpy as np

SPEC=C.BASE/'protocols/eyeball_omi_engineering_retry16k_v1.json'
OUT=C.BASE/'results/eyeball_omi_pilot_retry16k_v1'


def context(out,create=False):
    spec=C.read(SPEC)
    candidate=dict(protocol=spec,environment=C.environment(),code_sha256={**C.source_hashes(),
        str(C.Path(__file__).relative_to(C.ROOT)):C.sha(__file__),str(SPEC.relative_to(C.ROOT)):C.sha(SPEC)},
        parent_manifest_sha256=C.sha(C.BASE/'results/eyeball_omi_pilot_v1/manifest.json'),
        diagnostic_manifest_sha256=C.sha(C.BASE/'results/eyeball_cap_17086_v1/manifest.json'),
        splits_sha256=C.sha(C.SOURCE/'splits.csv'))
    p=out/'manifest.json'
    if p.exists():
        if C.read(p)!=candidate or C.sha(p)!=(out/'manifest.sha256').read_text().strip():
            raise ValueError('Changed retry context; use another version')
    elif create:
        C.write(p,candidate)
        C.put(out/'manifest.sha256',(C.sha(p)+'\n').encode())
        for name in candidate['code_sha256']:
            C.put(out/'source'/name,(C.ROOT/name).read_bytes())
        for name in ['splits.csv','source_records.csv']:
            C.put(out/name,(C.SOURCE/name).read_bytes())
    else: raise ValueError('Retry has not been prepared')
    return spec,C.sha(p)


def run(out=OUT):
    started=time.perf_counter()
    with C.lock(out):
        spec,digest=context(out,create=True)
        if C.verified(out,digest):
            print('Verified completed retry; no recomputation',flush=True);return
        rows=C.selection(C.split_rows(),32,spec['pilot']['selection_id'])
        with (C.BASE/'results/eyeball_omi_pilot_v1/selection.csv').open(newline='') as stream:
            if list(C.csv.DictReader(stream))!=rows: raise ValueError('Original pilot sample changed')
        C.csv_write(out/'selection.csv',rows)
        C.event(out,'started',context_sha=digest)
        C.write(out/'status.json',dict(status='running',stage='load_verified_inputs',time_utc=C.now()))
        try:
            signals={}
            with ACSDataset(target='OMI') as ds:
                for row in rows:
                    record=ds.load_record(row['record_id'])
                    if record.info.patient_id!=row['patient_id'] or record.decision.waveform_sha256!=row['waveform_sha256']:
                        raise ValueError('Changed fit record identity')
                    signals[row['record_id']]=record.signal_mV.T.copy()
            for stage,leads in [('baseline_lead_I',['I']),('baseline_other_leads',LEADS[1:])]:
                C.write(out/'status.json',dict(status='running',stage=stage,time_utc=C.now()))
                for i,row in enumerate(rows):
                    rid=row['record_id']
                    for lead in leads:
                        one(out/'records'/rid/lead,signals[rid][LEADS.index(lead)],spec,digest,
                            dict(record_id=rid,patient_id=row['patient_id'],lead=lead,partition='fit'),retain=lead=='I')
                    print(f'{stage}: {i+1}/{len(rows)} ECGs',flush=True)
            for row in rows:
                rid=row['record_id']
                saved=C.arrays(out/'records'/rid/'I'/'features.npz')
                repeated,_,retained=extract(signals[rid][0],500,spec,retain=True)
                np.testing.assert_array_equal(repeated,saved['features'])
                np.testing.assert_array_equal(retained['modes'],saved['modes'])
            C.write(out/'status.json',dict(status='running',stage='perturbations',time_utc=C.now()))
            comparisons=[]
            for i,row in enumerate(rows):
                rid=row['record_id'];baseline=C.arrays(out/'records'/rid/'I'/'features.npz')['features']
                for name,x in perturbations(signals[rid][0],500,spec['pilot']['perturbation_seed']+i).items():
                    try:
                        features,_=one(out/'perturbations'/rid/name,x,spec,digest,
                            dict(record_id=rid,patient_id=row['patient_id'],lead='I',perturbation=name,seed=spec['pilot']['perturbation_seed']+i))
                        delta=2*np.abs(features[:10]-baseline[:10])/(np.abs(features[:10])+np.abs(baseline[:10])+1e-12)
                        shift=float(np.linalg.norm(features[10:]-baseline[10:])/max(baseline[9],1e-12))
                        comparisons.append(dict(record_id=rid,perturbation=name,status='complete',
                            symmetric_relative_changes=delta.tolist(),centroid_shift_over_envelope=shift,
                            warnings=int((delta>spec['pilot']['relative_feature_warning']).sum()),centroid_warning=shift>.1))
                    except RotationalError as exc:
                        comparisons.append(dict(record_id=rid,perturbation=name,status='failed',error=str(exc)))
                print(f'Perturbations: {i+1}/{len(rows)} ECGs',flush=True)
            details=[C.read(out/'records'/r['record_id']/lead/'details.json') for r in rows for lead in LEADS]
            elapsed=sum(d['total_seconds'] for d in details)
            summary=dict(status='complete',baseline_ECGs=len(rows),baseline_leads=len(details),
                exact_repeat_checks=len(rows),perturbation_attempts=len(comparisons),
                perturbation_failures=sum(c['status']=='failed' for c in comparisons),
                perturbation_feature_warnings=sum(c.get('warnings',0) for c in comparisons),
                perturbation_centroid_warnings=sum(c.get('centroid_warning',False) for c in comparisons),
                summed_baseline_extraction_seconds=elapsed,mean_seconds_per_ECG=elapsed/len(rows),
                max_sifts=max(s['sifts'] for d in details for s in d['sifting']),
                maximum_reconstruction_error_mV=max(d['reconstruction_max_abs_mV'] for d in details),
                seconds_this_session=time.perf_counter()-started,models_trained=0,official_test_processed=False)
            C.write(out/'summary.json',summary);C.write(out/'perturbations.json',comparisons)
            make_plots(out,rows)
            files=['selection.csv','summary.json','perturbations.json','pilot_geometry.png','splits.csv','source_records.csv']
            files += [str(p.relative_to(out)) for directory in ('records','perturbations')
                for p in sorted((out/directory).rglob('*')) if p.is_file()]
            C.complete(out,files,digest)
            C.write(out/'status.json',summary)
            C.event(out,'complete',**{k:v for k,v in summary.items() if k!='status'})
            print(C.json.dumps(summary,indent=2),flush=True)
        except BaseException:
            failure=dict(status='failed',traceback=traceback.format_exc(),time_utc=C.now())
            C.write(out/'status.json',failure);C.event(out,'failed',traceback=failure['traceback'])
            raise


if __name__=='__main__': run()
