"""ECG-09 all-window representations. Separate outputs; baseline is read-only."""
import argparse, csv, hashlib, importlib.util, json, os, platform, sys, time, traceback
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent; REPO=BASE.parents[2]; ATT=REPO/'attention method'
for path in (REPO,ATT,ATT/'.venv/torch-cu128'): sys.path.insert(0,str(path))
from attention_ecg.ecgdata import file_hash,array_hash,read_json,atomic_json,atomic_npz,writer_lock
from attention_ecg.ecgdata_training import (partition,fit_control,fit_swin,verify_trial,complete_trial,
                                            save_checkpoint,load_checkpoint,batch_indices)
from attention_ecg.ecgdata_report import collect_predictions,confusion,scores
from attention_ecg.cepstral import CepstralConfig,filterbank
from scipy.fft import dct
from scipy.signal import get_window
import torch
from torch import nn

spec=importlib.util.spec_from_file_location('ecg07_frozen',REPO/'experiments/ecgdata/07_lfcc_swin_all_windows/study.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
RUN=BASE/'results'; CACHE=BASE/'features'; REPORT=BASE/'reports'; SEEDS=(0,1,2)


def extract_bands(windows,cfg,batch=64):
    """Same numerical PSD/filterbank as cepstral_features, before DCT."""
    bank=filterbank(cfg); hann=get_window('hann',cfg.frame_samples,fftbins=True);out=[]
    for first in range(0,len(windows),batch):
        x=np.asarray(windows[first:first+batch],dtype=np.float64)
        pad=(-(x.shape[1]-cfg.frame_samples))%cfg.hop_samples
        if pad: x=np.pad(x,((0,0),(0,pad)),mode='reflect')
        frames=np.lib.stride_tricks.sliding_window_view(x,cfg.frame_samples,axis=1)[:,::cfg.hop_samples,:]
        spectrum=np.fft.rfft(frames*hann,n=cfg.n_fft,axis=-1)
        power=abs(spectrum)**2/(cfg.fs*np.sum(hann**2));power[...,1:-1 if cfg.n_fft%2==0 else None]*=2
        energies=power@bank.T
        out.append(np.log(np.maximum(energies,cfg.power_floor))[:,:,None,:].astype(np.float32))
    result=np.concatenate(out)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite band features')
    return result

class RawCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,16,7,stride=2,padding=3),nn.GELU(),
          nn.Conv1d(16,32,5,stride=2,padding=2),nn.GELU(),nn.Conv1d(32,64,5,stride=2,padding=2),nn.GELU(),
          nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(64,3))
    def forward(self,x): return self.net(x)

def setup(device):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if device=='cuda':
        if not torch.cuda.is_available() or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8': raise ValueError('CUDA unavailable or determinism setting missing')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False

def sources():
    paths=[Path(__file__),BASE/'PROTOCOL.md',old.PROTOCOL,Path(old.__file__),ATT/'attention_ecg/ecgdata.py',ATT/'attention_ecg/ecgdata_training.py',ATT/'attention_ecg/ecgdata_report.py',ATT/'attention_ecg/cepstral.py',ATT/'attention_ecg/model.py',REPO/'ecgvmd/data.py',REPO/'ecgvmd/segment.py',REPO/'ecgvmd/config.py',REPO/'ecgvmd/ecgdata_subjects.csv',REPO/'ecgvmd/ecgdata_subjects.json']
    return {str(p.relative_to(REPO)):file_hash(p) for p in paths}

def prepare():
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'completed.json').exists():
        saved=read_json(CACHE/'manifest.json')
        if saved['sources']!=sources() or saved['baseline_cache_sha256']!=file_hash(old.CACHE/'completed.json') or saved['baseline_run_sha256']!=file_hash(old.RUN/'completed.json') or not verify_trial(CACHE): raise ValueError('Changed completed feature context')
        print('Verified and reused ECG-09 features',flush=True);return
    data=old.verify_cache();p=old.protocol();windows,arrays,v1=old.build(p)
    for key in ('y','patients','fold','row_ids','window_start','v1_index'): np.testing.assert_array_equal(data[key],arrays[key])
    manifest=dict(sources=sources(),baseline_cache_sha256=file_hash(old.CACHE/'completed.json'),baseline_run_sha256=file_hash(old.RUN/'completed.json'),windows_sha256=array_hash(windows),python=platform.python_version(),numpy=np.__version__)
    path=CACHE/'manifest.json'
    if path.exists() and read_json(path)!=manifest: raise ValueError('Cache manifest mismatch')
    atomic_json(path,manifest)
    if (CACHE/'completed.json').exists():
        if not verify_trial(CACHE): raise ValueError('Corrupt completed features')
        print('Verified and reused ECG-09 features',flush=True);return
    start=time.perf_counter();cfg=CepstralConfig(**p['cepstral']); bands=extract_bands(windows,cfg)
    recovered=dct(bands.astype(np.float64),type=2,norm='ortho',axis=-1)[...,:16].astype(np.float32)
    error=float(np.max(abs(recovered-data['X'])));print('Max original LFCC reconstruction error',error,flush=True)
    if error>2e-5: raise ValueError('Log-band/LFCC consistency failure')
    atomic_npz(CACHE/'arrays.npz',bands=bands,raw=windows.astype(np.float32),**{k:arrays[k] for k in arrays})
    atomic_json(CACHE/'extraction.json',dict(seconds=time.perf_counter()-start,shape=list(bands.shape),lfcc_max_error=error,exclusions=0,started_windows=len(windows)))
    complete_trial(CACHE,['manifest.json','arrays.npz','extraction.json']);print('Feature extraction complete',flush=True)

def raw_fit(folder,data,fold,seed,p,device):
    if verify_trial(folder): return
    folder.mkdir(parents=True,exist_ok=True);tr,te=partition(data,fold);x=data['raw'];y=data['y'];s=p['training']
    torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    model=RawCNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=s['lr'],weight_decay=s['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=s['epochs'],eta_min=s['min_lr'])
    weights=len(tr)/(3*np.bincount(y[tr],minlength=3));lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    history=[];first=0;prior=0.;state=load_checkpoint(folder)
    if state:
        if state['seed']!=seed or state['fold']!=fold: raise ValueError('Checkpoint identity changed')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);history=state['history'];first=state['epoch'];prior=state['seconds'];torch.set_rng_state(state['rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    start=time.perf_counter()
    for epoch in range(first,s['epochs']):
        begin=time.perf_counter();model.train();losses=[];masses=[];lr=opt.param_groups[0]['lr']
        for b in batch_indices(len(tr),s['batch_size'],seed,epoch,s['shuffle_seed_offset']):
            ix=tr[b];opt.zero_grad(set_to_none=True);inputs=torch.from_numpy(x[ix,None,:]).to(device);target=torch.from_numpy(y[ix]).to(device);loss=lossfn(model(inputs),target)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),s['gradient_clip_norm'],error_if_nonfinite=True);opt.step();losses.append(float(loss.detach().cpu()));masses.append(float(weights[y[ix]].sum()))
        scheduler.step();history.append(dict(epoch=epoch+1,loss=float(np.average(losses,weights=masses)),lr=lr,seconds=time.perf_counter()-begin))
        save_checkpoint(folder,dict(epoch=epoch+1,seed=seed,fold=fold,history=history,seconds=prior+time.perf_counter()-start,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []))
        atomic_json(folder/'history.json',history)
    model.eval();probs=[]
    with torch.inference_mode():
        for i in range(0,len(te),s['batch_size']): probs.append(model(torch.from_numpy(x[te[i:i+s['batch_size']],None,:]).to(device)).softmax(-1).cpu().numpy())
    probs=np.concatenate(probs);assert np.isfinite(probs).all()
    atomic_npz(folder/'predictions.npz',indices=te,y=y[te],patients=data['patients'][te],probabilities=probs,pred=probs.argmax(1))
    atomic_json(folder/'fit.json',dict(seed=seed,fold=fold,train_indices=tr.tolist(),test_indices=te.tolist(),epochs=s['epochs'],class_weights=weights.tolist(),seconds=prior+time.perf_counter()-start,parameters=sum(x.numel() for x in model.parameters()),warnings=[]))
    complete_trial(folder,['checkpoint.pt','checkpoint.json','history.json','predictions.npz','fit.json'])

def read_data():
    if not verify_trial(CACHE): raise ValueError('Missing verified cache')
    with np.load(CACHE/'arrays.npz',allow_pickle=False) as z: return {k:z[k] for k in z.files}

def train(device):
    setup(device);prepare();data=read_data();p=old.protocol();p['model']['n_ceps']=32
    data['X']=data['bands'];RUN.mkdir(parents=True,exist_ok=True)
    manifest=dict(sources=sources(),feature_completed_sha256=file_hash(CACHE/'completed.json'),protocol=p,device=device,device_name=torch.cuda.get_device_name() if device=='cuda' else platform.processor(),python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda)
    with writer_lock(RUN):
        path=RUN/'manifest.json'
        if path.exists() and read_json(path)!=manifest: raise ValueError('Run manifest changed')
        atomic_json(path,manifest)
        if (RUN/'completed.json').exists():
            if not verify_trial(RUN):raise ValueError('Changed completed run')
            print('Completed run verified; no retraining',flush=True);return
        atomic_json(RUN/'status.json',dict(status='running',pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        try:
            for fold in range(5):
                jobs=[('band_logistic',None),*[(a,s) for a in ('band_swin','raw_cnn') for s in SEEDS]]
                for arm,seed in jobs:
                    name=f'{arm}_fold{fold}' if seed is None else f'{arm}_seed{seed}_fold{fold}';folder=RUN/name;t=time.perf_counter()
                    if arm=='band_logistic':fit_control(folder,data,fold,p)
                    elif arm=='band_swin':fit_swin(folder,data,fold,seed,p,device)
                    else:raw_fit(folder,data,fold,seed,p,device)
                    print(f'{name}: verified/completed in {time.perf_counter()-t:.1f}s',flush=True)
                    atomic_json(RUN/'status.json',dict(status='running',last_completed=name,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            complete_trial(RUN,['manifest.json']+[str(x.relative_to(RUN)) for x in sorted(RUN.glob('*/completed.json'))]);atomic_json(RUN/'status.json',dict(status='complete',trials=35))
        except BaseException:
            atomic_json(RUN/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));atomic_json(RUN/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise
    report()

def report():
    if not verify_trial(RUN): raise ValueError('Incomplete run')
    data=read_data();original=old.verify_cache();p=old.protocol();baseline=collect_predictions(old.RUN,original,p)
    arms={'lfcc_swin':baseline['swin'],'lfcc_logistic':baseline['logistic']}
    for arm in ('band_swin','band_logistic','raw_cnn'):
        predictions=[]
        for seed in ([None] if arm=='band_logistic' else SEEDS):
            out=np.full(len(data['y']),-1,dtype=int)
            for fold in range(5):
                name=f'{arm}_fold{fold}' if seed is None else f'{arm}_seed{seed}_fold{fold}';folder=RUN/name
                if not verify_trial(folder): raise ValueError('Missing fit '+name)
                with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                    expected=np.flatnonzero(data['fold']==fold);np.testing.assert_array_equal(z['indices'],expected);np.testing.assert_array_equal(z['patients'],data['patients'][expected]);np.testing.assert_array_equal(z['y'],data['y'][expected]);out[expected]=z['pred']
            if np.any(out<0): raise ValueError('Missing OOF predictions')
            predictions.append(out)
        arms[arm]=predictions
    sub=np.flatnonzero(data['v1_index']>=0);pos=sub[np.argsort(data['v1_index'][sub])]
    sets={'v1_windows':pos,'all_windows':np.arange(len(data['y']))}; pairs=[('band_swin','lfcc_swin'),('raw_cnn','lfcc_swin'),('band_swin','band_logistic'),('band_logistic','lfcc_logistic')];results={}
    for name,idx in sets.items():
        these={a:[v[idx] for v in pred] for a,pred in arms.items()};summ={a:dict(per_seed=[old.summarize(data['y'][idx],data['patients'][idx],v) for v in preds]) for a,preds in these.items()}
        for v in summ.values():v['mean']=old.mean_metrics(v['per_seed'])
        results[name]=dict(arms=summ,bootstrap=old.paired_bootstrap(data['y'][idx],data['patients'][idx],these,pairs,2000,20260925))
    fits=[]
    for f in sorted(RUN.glob('*/fit.json')):
        v=read_json(f);fits.append(dict(trial=f.parent.name,seconds=v['seconds'],warnings=v.get('warnings',[]),epochs=v.get('epochs')))
    analysis=dict(classes=p['classes'],seeds=list(SEEDS),results=results,fits=fits,source_manifest_sha256=file_hash(RUN/'manifest.json'),feature_audit=read_json(CACHE/'extraction.json'),uncertainty_scope='pointwise paired patient bootstrap; conditional on saved fits; no retraining uncertainty')
    REPORT.mkdir(exist_ok=True);atomic_json(REPORT/'analysis.json',analysis)
    atomic_npz(RUN/'predictions_export.npz',y=data['y'],patients=data['patients'],fold=data['fold'],row_ids=data['row_ids'],window_start=data['window_start'],v1_index=data['v1_index'],**{f'{a}_seed{i}':v for a,vs in arms.items() for i,v in enumerate(vs)})
    lines=['# ECG-09 representation comparison','','Primary evaluation: identical 1,620 v1 windows; values mean across seeds 0/1/2 except deterministic logistic.','','| Arm | Window accuracy | Window macro-F1 | Patient-vote accuracy | ARR / CHF / NSR recall |','|---|---:|---:|---:|---|']
    for a,v in results['v1_windows']['arms'].items():
        w=v['mean']['window'];pt=v['mean']['patient_vote'];r=w['sensitivity'];lines.append(f'| {a} | {w["accuracy"]:.2%} | {w["macro_f1"]:.4f} | {pt["accuracy"]:.2%} | {r[0]:.2%} / {r[1]:.2%} / {r[2]:.2%} |')
    for a,b in pairs:
        left=results['v1_windows']['arms'][a]['mean']['window']['macro_f1'];right=results['v1_windows']['arms'][b]['mean']['window']['macro_f1'];ci=results['v1_windows']['bootstrap']['paired_differences'][f'{a}_minus_{b}']['window_macro_f1']['interval_95'];lines.append(f'\n{a} minus {b}: macro-F1 {left-right:+.4f}, paired 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}].')
    lines+=['','All-window results, all seeds, class recalls and uncertainty: [analysis.json](analysis.json).',f'35 new fits; summed fit runtime {sum(f["seconds"] for f in fits):.1f} s; no data exclusions.', 'Pointwise bootstrap intervals condition on saved fits and do not include retraining uncertainty. Different representation input dimensions and raw CNN capacity are disclosed in PROTOCOL.md. ECGData contains only 80 patients and source/diagnosis confounding; no external or clinical validation.']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','train','report'));parser.add_argument('--execute-training',action='store_true');parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='train':
        if not args.execute_training:raise ValueError('Pass --execute-training')
        train(args.device)
    else:report()
if __name__=='__main__':main()
