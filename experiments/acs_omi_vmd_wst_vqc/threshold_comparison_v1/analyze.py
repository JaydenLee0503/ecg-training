"""Fixed-threshold validation analysis of completed OMI model scores."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;REPO=BASE.parents[2];sys.path.insert(0,str(REPO))
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.common import read,sha,arrays,verified,write,npz
from experiments.acs_omi_vmd_wst_vqc.sensitivity_study_v1.metrics import Curve,at_threshold
SRC=BASE.parent/'undersampled_v1'; IMP=BASE.parent/'omi_improvement_v2'; PARENT=BASE.parent/'sensitivity_study_v1/results/models/lfcc';OUT=BASE/'results'
FIELDS=('sensitivity','specificity','precision','tp','fp','fn','tn')

def main():
 if not verified(SRC/'results'):raise ValueError('Missing completed undersampled run')
 data=arrays(IMP/'results/prepared/cohort.npz');idx=np.flatnonzero(data['partitions']=='validation');cal=np.flatnonzero(data['partitions']=='calibration'); y=data['y'][idx];patients=data['patients'][idx];unique,inverse=np.unique(patients,return_inverse=True)
 print(f'Validation ECGs={len(y)} patients={len(unique)} positives={y.sum()}',flush=True)
 saved=read(SRC/'analysis.json');scores={};thresholds={}
 for kind in ('undersampled','oversampled'):
  for arm,seed in [('logistic',None)]+[('swin',i) for i in range(3)]:
   name=kind+'_'+(arm if seed is None else f'{arm}_seed{seed}')
   if kind=='undersampled':
    folder=SRC/'results'/('logistic' if seed is None else f'swin_seed{seed}');z=arrays(folder/'validation.npz');cz=arrays(folder/('held.npz' if seed is None else 'epoch40/held.npz'));points=read(folder/'thresholds.json')
   else:
    folder=PARENT/('logistic' if seed is None else f'swin_seed{seed}');
    if not verified(folder): raise ValueError('Missing oversampled baseline')
    z=arrays(folder/'validation_predictions.npz');cz=arrays(folder/'calibration_predictions.npz');points=read(folder/'thresholds.json')['calibration_points']
   for k in ('record_ids','patients','y'):
    np.testing.assert_array_equal(z[k],data[k][idx]);np.testing.assert_array_equal(cz[k],data[k][cal])
   if Curve(cz['y'],cz['score']).operating_points([80,90,95])!=points:raise ValueError('Calibration cutoff changed')
   scores[name]=z['score'];thresholds[name]=[.5]+[p['threshold'] for p in points]
 groups={kind+'_'+arm:[kind+'_'+arm] if arm=='logistic' else [kind+'_swin_seed'+str(i) for i in range(3)] for kind in ('undersampled','oversampled') for arm in ('logistic','swin')}
 grouped={}
 for name,score in scores.items():
  grouped[name]=[]
  for threshold in thresholds[name]:
   pred=score>=threshold
   grouped[name].append(np.stack([np.bincount(inverse,weights=((y==1)&pred),minlength=len(unique)),np.bincount(inverse,weights=((y==0)&pred),minlength=len(unique)),np.bincount(inverse,weights=((y==1)&~pred),minlength=len(unique)),np.bincount(inverse,weights=((y==0)&~pred),minlength=len(unique))],axis=1))
  grouped[name]=np.stack(grouped[name])
 def stats(name,multiplicity):
  c=(grouped[name]*multiplicity[None,:,None]).sum(axis=1);tp,fp,fn,tn=c.T
  return np.stack([tp/(tp+fn),tn/(tn+fp),np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0),tp,fp,fn,tn],axis=1)
 def evaluate(multiplicity):return {g:np.mean([stats(m,multiplicity) for m in members],axis=0) for g,members in groups.items()}
 point=evaluate(np.ones(len(unique)));draws={g:[] for g in groups};rng=np.random.default_rng(20261101);rejected=0
 for _ in range(2000):
  mult=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique));
  if not ((mult[inverse][y==1].sum()>0) and (mult[inverse][y==0].sum()>0)):rejected+=1;continue
  for g,v in evaluate(mult).items():draws[g].append(v)
 OUT.mkdir(exist_ok=True);npz(OUT/'bootstrap.npz',**{k:np.asarray(v) for k,v in draws.items()})
 summary={'threshold_labels':['fixed_0.5','cal_target_80','cal_target_90','cal_target_95'],'seeds':[0,1,2],'bootstrap':{'requested':2000,'valid':2000-rejected,'seed':20261101},'results':{},'comparisons':{},'sources':{'protocol_sha256':sha(BASE/'PROTOCOL.md'),'analysis_sha256':sha(Path(__file__)),'source_completion_sha256':sha(SRC/'results/completed.json'),'source_analysis_sha256':sha(SRC/'analysis.json')}}
 for g,v in point.items():
  arr=np.asarray(draws[g]);lo,hi=np.quantile(arr,[.025,.975],axis=0);summary['results'][g]=[dict(point=dict(zip(FIELDS,v[i].tolist())),intervals={k:[float(lo[i,j]),float(hi[i,j])] for j,k in enumerate(FIELDS)}) for i in range(4)]
 for arm in ('logistic','swin'):
  a='undersampled_'+arm;b='oversampled_'+arm;diff=point[a]-point[b];d=np.asarray(draws[a])-np.asarray(draws[b]);lo,hi=np.quantile(d,[.025,.975],axis=0);summary['comparisons'][arm]=[dict(point=dict(zip(FIELDS,diff[i].tolist())),intervals={k:[float(lo[i,j]),float(hi[i,j])] for j,k in enumerate(FIELDS)}) for i in range(4)]
 write(BASE/'analysis.json',summary)
 lines=['# OMI fixed and calibration-selected threshold comparison','','These are fixed previously selected thresholds; calibration targets do not guarantee validation sensitivity. All Swin values are means over seeds 0/1/2.','','| Model | Cutoff | Sensitivity | Specificity | Precision | False positives |','|---|---|---:|---:|---:|---:|']
 for g,rows in summary['results'].items():
  for label,row in zip(summary['threshold_labels'],rows):
   v=row['point'];lines.append(f'| {g} | {label} | {v["sensitivity"]:.2%} | {v["specificity"]:.2%} | {v["precision"]:.2%} | {v["fp"]:.1f} |')
 lines+=['','2,000 paired patient-bootstrap draws; fixed previously selected thresholds. Intervals and paired differences are in [analysis.json](analysis.json). The same validation cohort has been repeatedly studied; comparison is internal and exploratory. Test ECGs untouched.']
 (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n');write(OUT/'completed.json',{'status':'complete','protocol_sha256':sha(BASE/'PROTOCOL.md'),'source_sha256':sha(Path(__file__)),'report_sha256':sha(BASE/'REPORT.md'),'analysis_sha256':sha(BASE/'analysis.json'),'draws_sha256':sha(OUT/'bootstrap.npz'),'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
 print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
