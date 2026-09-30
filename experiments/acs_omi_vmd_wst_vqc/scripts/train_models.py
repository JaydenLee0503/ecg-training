"""Fixed ACS development comparison; explicit preparation and resumable fitting."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from experiments.acs_omi_vmd_wst_vqc.scripts import extract_gpu_retry as retry
from experiments.acs_omi_vmd_wst_vqc.training import CheckpointVQC, atomic_joblib
from experiments.acs_omi_vmd_wst_vqc.pipeline import fit_preprocessor, make_classifiers
from experiments.acs_omi_vmd_wst_vqc.scripts.bundle_gpu import read_extraction_run
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

BASE=retry.BASE
ROOT=retry.ROOT
old=retry.old
TRIALS=[(arm,name) for arm in ('vmd','wst') for name in
        ('logistic','weighted_knn','vqc_seed0','vqc_seed1','vqc_seed2')]
CODE=['training.py','reporting.py','scripts/train_models.py','scripts/report_models.py','scripts/bundle_gpu.py',
      'tests/test_training.py','tests/test_reporting.py','tests/test_training_runner.py','tests/test_bundle.py']


def code_hashes():
    paths=[BASE/name for name in CODE]+[ROOT/'ecgvmd/reupload.py',ROOT/'ecgvmd/select.py',
                                       ROOT/'ecgvmd/quantum.py',BASE/'pipeline.py']
    return {str(p.relative_to(ROOT)):old.serial.file_hash(p) for p in paths}


def atomic_npz(path,**arrays):
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    path.with_suffix('.tmp').replace(path)


def checked_files(folder,marker,context_sha):
    if marker['status']!='complete' or marker['context_sha']!=context_sha:
        raise ValueError('Incomplete or mismatched saved result')
    for name,digest in marker['files'].items():
        if old.serial.file_hash(folder/name)!=digest:
            raise ValueError(f'Changed saved artifact: {folder/name}')


def read_run(out):
    sha=old.serial.file_hash(out/'manifest.json')
    if (out/'manifest.sha256').read_text().strip()!=sha:
        raise ValueError('Changed training manifest')
    m=json.loads((out/'manifest.json').read_text())
    if m['code_sha256']!=code_hashes() or m['environment']!=old.environment():
        raise ValueError('Changed training implementation/environment; use a separate run')
    source=ROOT/m['source_run']
    if old.serial.file_hash(source/'manifest.json')!=m['source_manifest_sha256']:
        raise ValueError('Changed feature source manifest')
    return m,sha


def prepare(source,out):
    m,rows,source_sha=read_extraction_run(source)
    bundle=json.loads((source/'bundle.json').read_text())
    if (bundle['status']!='complete' or bundle['manifest_sha256']!=source_sha
            or bundle['features_sha256']!=old.serial.file_hash(source/'features.npz')):
        raise ValueError('Need a verified complete feature bundle')
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(source_run=str(source.resolve().relative_to(ROOT)),source_manifest_sha256=source_sha,
        features_sha256=bundle['features_sha256'],protocol=m['scientific_protocol'],
        code_sha256=code_hashes(),environment=old.environment(),trials=TRIALS,
        evaluation='Fixed internal validation; no threshold/epoch/model selection; official test reserved')
    with old.run_lock(out):
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())!=json.loads(json.dumps(manifest)):
                raise ValueError('Changed training run; choose a new output')
        else:
            old.atomic_json(out/'manifest.json',manifest)
            old.atomic_bytes(out/'manifest.sha256',(old.serial.file_hash(out/'manifest.json')+'\n').encode())
        sha=old.serial.file_hash(out/'manifest.json')
        snapshots=out/'source'
        snapshots.mkdir(exist_ok=True)
        for name,digest in manifest['code_sha256'].items():
            target=snapshots/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if old.serial.file_hash(target)!=digest:
                    raise ValueError('Changed saved training source snapshot')
            else:
                old.atomic_bytes(target,(ROOT/name).read_bytes())
        with np.load(source/'features.npz',allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        if (str(data['manifest_sha256'])!=source_sha
                or list(data['record_ids'])!=[r['record_id'] for r in rows]
                or list(data['patients'])!=[r['patient_id'] for r in rows]
                or list(data['partitions'])!=[r['partition'] for r in rows]
                or list(data['y'])!=[int(r['label']) for r in rows]):
            raise ValueError('Feature row identities do not match the frozen split')
        for arm in ('vmd','wst'):
            folder=out/arm
            folder.mkdir(exist_ok=True)
            marker=folder/'preprocessing.json'
            if marker.exists():
                checked_files(folder,json.loads(marker.read_text()),sha)
                continue
            started=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                preprocessor,angles,fit,val=fit_preprocessor(data[arm],data['y'],data['patients'],
                                                           data['partitions'],manifest['protocol'])
            selected=preprocessor[0].idx_
            np.testing.assert_allclose(preprocessor[1].scaler_.mean_,
                data[arm][fit][:,selected].astype(float).mean(0),atol=1e-12,rtol=0)
            atomic_joblib(folder/'preprocessor.joblib',preprocessor)
            atomic_npz(folder/'inputs.npz',X_fit=angles[fit],X_validation=angles[val],
                y_fit=data['y'][fit],y_validation=data['y'][val],
                fit_record_ids=data['record_ids'][fit],validation_record_ids=data['record_ids'][val],
                fit_patients=data['patients'][fit],validation_patients=data['patients'][val],
                selected_indices=selected,selected_names=data[arm+'_names'][selected])
            old.atomic_json(marker,dict(status='complete',context_sha=sha,
                files={n:old.serial.file_hash(folder/n) for n in ('preprocessor.joblib','inputs.npz')},
                seconds=time.perf_counter()-started,warnings=[str(w.message) for w in caught],
                fit_records=len(fit),validation_records=len(val),selected_names=data[arm+'_names'][selected].tolist()))
            print(f'Prepared {arm}: {len(fit)} fit, {len(val)} validation ECGs',flush=True)
        old.atomic_json(out/'preparation.json',dict(status='complete',context_sha=sha,completed_at_utc=old.serial.now()))


def trial(out,arm,name):
    out=Path(out)
    m,sha=read_run(out)
    parent=out/arm
    checked_files(parent,json.loads((parent/'preprocessing.json').read_text()),sha)
    folder=parent/name
    folder.mkdir(exist_ok=True)
    with old.run_lock(folder),threadpool_limits(limits=1):
        if (folder/'completed.json').exists():
            checked_files(folder,json.loads((folder/'completed.json').read_text()),sha)
            return f'{arm}/{name}: verified complete'
        with np.load(parent/'inputs.npz',allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        started=time.perf_counter()
        old.atomic_json(folder/'status.json',dict(status='running',started_at_utc=old.serial.now(),context_sha=sha))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if name.startswith('vqc_seed'):
                    settings={k:v for k,v in m['protocol']['vqc'].items() if k not in ('class','seeds')}
                    model=CheckpointVQC(seed=int(name[-1]),**settings)
                    model.fit(data['X_fit'],data['y_fit'],folder/'epochs',sha)
                    old.atomic_json(folder/'history.json',model.history_)
                    model.save_weights(folder/'weights.npz')
                else:
                    model=make_classifiers(m['protocol'])[name]
                    model.fit(data['X_fit'],data['y_fit'])
                if list(model.classes_)!=[0,1]:
                    raise ValueError('Unexpected probability class order')
                score=model.predict_proba(data['X_validation'])[:,1]
                if not np.isfinite(score).all() or np.any((score<0)|(score>1)):
                    raise ValueError('Invalid validation probabilities')
            pipeline=Pipeline([('preprocessor',joblib.load(parent/'preprocessor.joblib')),('classifier',model)])
            atomic_joblib(folder/'model.joblib',pipeline)
            restored=joblib.load(folder/'model.joblib')['classifier']
            np.testing.assert_array_equal(restored.predict_proba(data['X_validation'])[:,1],score)
            atomic_npz(folder/'predictions.npz',score=score,prediction=(score>=0.5).astype(int),
                y=data['y_validation'],record_ids=data['validation_record_ids'],patients=data['validation_patients'])
            files=['model.joblib','predictions.npz']
            if name.startswith('vqc_seed'):
                files+=['history.json','weights.npz']+[str(p.relative_to(folder)) for p in sorted((folder/'epochs').glob('*'))]
            record=dict(status='complete',context_sha=sha,arm=arm,classifier=name,
                seconds=time.perf_counter()-started,completed_at_utc=old.serial.now(),
                warnings=[str(w.message) for w in caught],
                files={n:old.serial.file_hash(folder/n) for n in files},
                training_seconds=model.history_[-1]['seconds'] if name.startswith('vqc_seed') else None)
            old.atomic_json(folder/'completed.json',record)
            old.atomic_json(folder/'status.json',record)
            return f'{arm}/{name}: complete ({record["seconds"]:.1f}s this session)'
        except BaseException:
            failure=dict(status='failed',context_sha=sha,traceback=traceback.format_exc(),time_utc=old.serial.now())
            old.atomic_json(folder/f'failure_{time.time_ns()}.json',failure)
            old.atomic_json(folder/'status.json',failure)
            raise


def train(out,jobs):
    read_run(out)
    with old.run_lock(out), ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(trial,str(out),arm,name) for arm,name in TRIALS]
        for future in futures:
            print(future.result(),flush=True)
        _,sha=read_run(out)
        old.atomic_json(out/'completed.json',dict(status='complete',context_sha=sha,fits=len(TRIALS),
            completed_at_utc=old.serial.now(),official_test_processed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train'))
    parser.add_argument('--source',type=Path,default=BASE/'results/omi_v1_gpu_retry128k')
    parser.add_argument('--out',type=Path,default=BASE/'results/omi_models_v1_retry128k')
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.jobs<=6:
        parser.error('jobs must be between 1 and 6')
    with threadpool_limits(limits=1):
        prepare(args.source,args.out) if args.command=='prepare' else train(args.out,args.jobs)
