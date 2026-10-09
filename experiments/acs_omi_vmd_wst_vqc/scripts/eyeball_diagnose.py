"""Bounded fit-only cap diagnosis; keep the stopped pilot and criteria intact."""
import copy
import time
import traceback
import warnings
from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset
from ecgvmd.rotational import extract, RotationalError, NAMES
import numpy as np


def run():
    out=C.BASE/'results/eyeball_cap_17086_v1'
    parent=C.BASE/'results/eyeball_omi_pilot_v1'
    limits=[4000,16000]
    spec=C.read(C.SPEC)
    row=next(r for r in C.split_rows() if r['record_id']=='17086')
    if row['partition']!='fit': raise ValueError('Diagnostic requires a fit patient')
    candidate=dict(record=row,lead='I',limits=limits,only_change='MAX_ITERATION',
        scientific_spec=spec,parent_manifest_sha256=C.sha(parent/'manifest.json'),
        code_sha256={**C.source_hashes(),str(C.Path(__file__).relative_to(C.ROOT)):C.sha(__file__)},
        environment=C.environment())
    with C.lock(out):
        if (out/'manifest.json').exists():
            if C.read(out/'manifest.json')!=candidate: raise ValueError('Changed diagnostic')
        else:
            C.write(out/'manifest.json',candidate)
            for name in candidate['code_sha256']:
                C.put(out/'source'/name,(C.ROOT/name).read_bytes())
        digest=C.sha(out/'manifest.json')
        if C.verified(out,digest):
            print('Verified completed cap diagnosis',flush=True);return
        results=[]
        with ACSDataset(target='OMI') as ds:
            record=ds.load_record(row['record_id'])
            if record.decision.waveform_sha256!=row['waveform_sha256']:
                raise ValueError('Changed diagnostic waveform')
            x=record.lead('I')
            for limit in limits:
                setting=copy.deepcopy(spec)
                setting['emd']['parameters']['MAX_ITERATION']=limit
                folder=out/str(limit)
                if C.verified(folder,digest):
                    results.append(C.read(folder/'details.json'));break
                C.event(out,'started',limit=limit,record_id=row['record_id'],lead='I')
                started=time.perf_counter()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        features,details,retained=extract(x,500,setting,retain=True)
                    # Separate uninstrumented package invocation checks hook neutrality.
                    from PyEMD import EMD
                    reference=EMD(**setting['emd']['parameters'])
                    reference.emd(x,max_imf=4)
                    modes,residual=reference.get_imfs_and_residue()
                    np.testing.assert_array_equal(modes,retained['modes'])
                    np.testing.assert_array_equal(residual,retained['residual'])
                    details.update(status='complete',limit=limit,reference_exact_match=True,
                        warnings=[str(w.message) for w in caught],seconds_with_reference=time.perf_counter()-started)
                    C.npz(folder/'features.npz',features=features,names=np.array(NAMES),**retained)
                    C.write(folder/'details.json',details)
                    C.complete(folder,['features.npz','details.json'],digest)
                    C.event(out,'complete',limit=limit,diagnostics=details)
                    results.append(details)
                    print(C.json.dumps(details,indent=2),flush=True)
                    break
                except RotationalError as exc:
                    details=dict(status='failed',limit=limit,error=str(exc),diagnostics=exc.diagnostics,
                        traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
                    C.write(folder/f'failure_{time.time_ns()}.json',details)
                    C.event(out,'failed',limit=limit,error=str(exc),diagnostics=exc.diagnostics)
                    results.append(details)
                    print(C.json.dumps(details,indent=2),flush=True)
        summary=dict(status='complete',results=results,resolved=any(r['status']=='complete' for r in results),
                     stopping_criteria_changed=False,models_trained=0,official_test_processed=False)
        C.write(out/'summary.json',summary)
        names=[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name!='.writer.lock']
        C.complete(out,names,digest)


if __name__=='__main__': run()
