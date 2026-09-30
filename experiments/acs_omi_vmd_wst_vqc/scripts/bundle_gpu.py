"""Verify and bundle a complete amended GPU run without reading waveforms."""
import argparse
import json
from pathlib import Path

from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu_retry as retry
from experiments.acs_omi_vmd_wst_vqc.gpu_retry import assert_converged
import numpy as np


def read_extraction_run(out):
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['id']=='acs-omi-v1-retry256k':
        from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu_retry256k
        return extract_gpu_retry256k.read_run(out)
    if protocol['id']=='acs-omi-v1-retry128k':
        return retry.read_run(out)
    raise ValueError('Unsupported extraction protocol; no implicit migration')


def bundle(out):
    m, rows, sha = read_extraction_run(out)
    old = retry.old
    with old.run_lock(out):
        status = json.loads((out/'extraction_status.json').read_text())
        if status['status'] != 'complete' or status['records_complete'] != len(rows) or status['manifest_sha256'] != sha:
            raise ValueError('Full extraction must complete before bundling')
        target = out/'features.npz'
        if (out/'bundle.json').exists():
            saved=json.loads((out/'bundle.json').read_text())
            if saved['manifest_sha256']!=sha or saved['features_sha256']!=old.serial.file_hash(target):
                raise ValueError('Changed completed feature bundle')
            print('Verified completed feature bundle',flush=True)
            return
        blocks={arm:np.empty((len(rows),dim),np.float32) for arm,dim in [('vmd',2688),('wst',2808)]}
        names={}
        max_iterations=0
        for i,row in enumerate(rows):
            vectors,meta=old.checked(out/'features',row,sha)
            assert_converged(meta,m['scientific_protocol'])
            max_iterations=max(max_iterations,max(meta['vmd_iterations']))
            for arm in blocks:
                if arm in names:
                    np.testing.assert_array_equal(names[arm],vectors[arm+'_names'])
                names[arm]=vectors[arm+'_names']
                blocks[arm][i]=vectors[arm]
            if (i+1)%1000==0:
                print(f'Verified {i+1}/{len(rows)} ECGs',flush=True)
        patients=np.array([r['patient_id'] for r in rows])
        partitions=np.array([r['partition'] for r in rows])
        if set(patients[partitions=='fit']) & set(patients[partitions=='validation']):
            raise ValueError('Patient overlap')
        with (out/'features.tmp').open('wb') as stream:
            np.savez_compressed(stream,**blocks,**{arm+'_names':n for arm,n in names.items()},
                y=np.array([int(r['label']) for r in rows]),patients=patients,partitions=partitions,
                record_ids=np.array([r['record_id'] for r in rows]),manifest_sha256=np.array(sha))
        (out/'features.tmp').replace(target)
        old.atomic_json(out/'bundle.json',dict(status='complete',records=len(rows),
            features_sha256=old.serial.file_hash(target),manifest_sha256=sha,max_vmd_iterations=max_iterations,
            final_capped_leads=0,dimensions={arm:list(x.shape) for arm,x in blocks.items()},
            completed_at_utc=old.serial.now(),official_test_processed=False))
        print(f'Bundled {len(rows)} verified development ECGs',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=retry.BASE/'results/omi_v1_gpu_retry128k')
    bundle(parser.parse_args().out)
