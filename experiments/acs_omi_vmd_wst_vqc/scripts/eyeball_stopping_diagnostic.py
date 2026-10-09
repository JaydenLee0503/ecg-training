"""One fit ECG, declared 64k default-criterion diagnosis and S-number comparison.

No labels or classifier scores select either setting. Both outcomes are retained.
This probes an alternative stopping algorithm, not an equivalent retry.
"""
import time
import warnings
from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset
from ecgvmd.rotational import analytic_descriptors, NAMES
from PyEMD import EMD
import numpy as np


def run():
    out=C.BASE/'results/eyeball_stopping_diagnostic_v1'
    spec=C.read(C.SPEC)
    row=next(r for r in C.split_rows() if r['record_id']=='15689')
    if row['partition']!='fit': raise ValueError('Fit data only')
    configurations=[dict(name='default_64000',MAX_ITERATION=64000,FIXE_H=0),
                    dict(name='s_number5_1000',MAX_ITERATION=1000,FIXE_H=5)]
    candidate=dict(record=row,lead='I',configurations=configurations,base_spec=spec,
        rationale='Diagnose cap using default criteria; separately assess standard consecutive-extrema/zero-crossing stop; no predictive scoring',
        environment=C.environment(),code_sha256={**C.source_hashes(),
            str(C.Path(__file__).relative_to(C.ROOT)):C.sha(__file__)})
    with C.lock(out):
        if (out/'manifest.json').exists():
            if C.read(out/'manifest.json')!=candidate: raise ValueError('Changed diagnostic')
        else:
            C.write(out/'manifest.json',candidate)
            for name in candidate['code_sha256']:
                C.put(out/'source'/name,(C.ROOT/name).read_bytes())
        digest=C.sha(out/'manifest.json')
        if C.verified(out,digest): print('Verified stopping diagnostic',flush=True);return
        with ACSDataset(target='OMI') as ds:
            record=ds.load_record(row['record_id'])
            if record.decision.waveform_sha256!=row['waveform_sha256']:raise ValueError('Waveform changed')
            x=record.lead('I')
        results=[]
        for config in configurations:
            folder=out/config['name']
            if C.verified(folder,digest):results.append(C.read(folder/'details.json'));continue
            params={**spec['emd']['parameters'],**{k:v for k,v in config.items() if k!='name'}}

            class Observed(EMD):
                def __init__(self):
                    super().__init__(**params)
                    self.logger=self;self.count=0;self.cap=False;self.records=[];self.checks=[]
                def debug(self,*args):pass
                def info(self,message,*args):
                    if message.startswith('Max iterations'):self.cap=True
                def extract_max_min_spline(self,*args):
                    self.count+=1
                    return super().extract_max_min_spline(*args)
                def check_imf(self,new,old,maximum,minimum):
                    result=super().check_imf(new,old,maximum,minimum)
                    if self.count in (100,1000,4000,16000,32000,63999):
                        e=self.find_extrema(np.arange(len(new),dtype=float),new)
                        delta=new-old
                        item=dict(sifts=self.count,criterion_pass=bool(result),
                            extrema=len(e[0])+len(e[2]),zero_crossings=len(e[4]),
                            extrema_sign_pass=bool(np.all(maximum[1]>=0) and np.all(minimum[1]<=0)),
                            energy_ratio=float(np.sum(delta**2)/np.sum(old**2)),
                            max_abs_change=float(np.max(np.abs(delta))))
                        self.checks.append(item)
                        print(config['name'],len(self.records)+1,item,flush=True)
                    return result
                def end_condition(self,S,IMF):
                    result=super().end_condition(S,IMF)
                    self.records.append(dict(candidate=len(IMF),sifts=self.count,capped=self.cap,checks=self.checks))
                    self.count=0;self.cap=False;self.checks=[]
                    return result

            C.event(out,'started',configuration=config)
            started=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                solver=Observed();solver.emd(x,max_imf=4)
                modes,residual=solver.get_imfs_and_residue()
            valid=len(modes)==4 and not any(r['capped'] for r in solver.records)
            details=dict(configuration=config,numerically_valid=valid,seconds=time.perf_counter()-started,
                components=len(modes),sifting=solver.records,warnings=[str(w.message) for w in caught],
                reconstruction_max_abs_mV=float(np.max(np.abs(x-modes.sum(0)-residual))))
            data=dict(modes=modes,residual=residual)
            if valid:
                values,diagnostics,trajectory=analytic_descriptors(modes,500,spec['analytic'])
                details['analytic']=diagnostics
                data.update(features=values,names=np.array(NAMES),trajectory=trajectory)
                original=EMD(**params);original.emd(x,max_imf=4)
                expected,remainder=original.get_imfs_and_residue()
                np.testing.assert_array_equal(modes,expected);np.testing.assert_array_equal(residual,remainder)
                details['uninstrumented_exact_match']=True
            C.write(folder/'details.json',details);C.npz(folder/'decomposition.npz',**data)
            C.complete(folder,['details.json','decomposition.npz'],digest)
            C.event(out,'complete',configuration=config,numerically_valid=valid)
            results.append(details)
            print(config['name'],'complete; valid=',valid,'seconds=',details['seconds'],flush=True)
        summary=dict(status='complete',results=results,models_trained=0,official_test_processed=False)
        C.write(out/'summary.json',summary)
        names=[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name!='.writer.lock']
        C.complete(out,names,digest)


if __name__=='__main__':run()
