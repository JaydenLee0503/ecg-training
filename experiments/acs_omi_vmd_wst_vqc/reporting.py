"""ECG metrics and paired resampling of whole validation patients."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS=('average_precision','roc_auc','balanced_accuracy','sensitivity','specificity',
         'precision','f1','accuracy')


def binary_metrics(y,score,weights=None):
    y,score=np.asarray(y),np.asarray(score)
    weights=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    if (y.ndim!=1 or score.shape!=y.shape or weights.shape!=y.shape
            or not np.isfinite(score).all() or np.any((score<0)|(score>1))
            or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0
            or not set(np.unique(y))<= {0,1}):
        raise ValueError('Invalid labels, probabilities or sample weights')
    pred=score>=0.5
    tp=float(weights[(y==1)&pred].sum()); fn=float(weights[(y==1)&~pred].sum())
    tn=float(weights[(y==0)&~pred].sum()); fp=float(weights[(y==0)&pred].sum())
    if tp+fn==0 or tn+fp==0:
        raise ValueError('Ranking metrics require both classes with positive weight')
    sensitivity=tp/(tp+fn); specificity=tn/(tn+fp)
    return dict(average_precision=float(average_precision_score(y,score,sample_weight=weights)),
        roc_auc=float(roc_auc_score(y,score,sample_weight=weights)),
        balanced_accuracy=(sensitivity+specificity)/2,sensitivity=sensitivity,specificity=specificity,
        precision=tp/(tp+fp) if tp+fp else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        accuracy=(tp+tn)/weights.sum(),tn=tn,fp=fp,fn=fn,tp=tp)


def patient_weights(patients,rng):
    unique,inverse=np.unique(patients,return_inverse=True)
    counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
    return counts[inverse]


def bootstrap(y,patients,scores,groups,pairs,resamples,seed):
    """Groups average seed metrics, never probabilities; all models share each draw."""
    point={name:binary_metrics(y,s) for name,s in scores.items()}
    group_point={name:{k:float(np.mean([point[member][k] for member in members])) for k in METRICS}
                 for name,members in groups.items()}
    draws={name:[] for name in groups}
    differences={name:[] for name in pairs}
    rng=np.random.default_rng(seed)
    skipped=0
    for i in range(resamples):
        weights=patient_weights(patients,rng)
        if not all(weights[np.asarray(y)==label].sum()>0 for label in (0,1)):
            skipped+=1
            continue
        metrics={name:binary_metrics(y,s,weights) for name,s in scores.items()}
        grouped={name:np.array([np.mean([metrics[member][k] for member in members]) for k in METRICS])
                 for name,members in groups.items()}
        for name,values in grouped.items():
            draws[name].append(values)
        for name,(left,right) in pairs.items():
            differences[name].append(grouped[left]-grouped[right])
        if (i+1)%200==0:
            print(f'Patient bootstrap: {i+1}/{resamples}',flush=True)
    if not resamples-skipped:
        raise ValueError('No two-class bootstrap draws')
    def intervals(values):
        low,high=np.percentile(np.asarray(values),[2.5,97.5],axis=0)
        return {k:[float(a),float(b)] for k,a,b in zip(METRICS,low,high)}
    return dict(per_model=point,groups=group_point,
        group_intervals={name:intervals(values) for name,values in draws.items()},
        paired_differences={name:dict(left=left,right=right,
            point={k:group_point[left][k]-group_point[right][k] for k in METRICS},
            intervals=intervals(differences[name])) for name,(left,right) in pairs.items()},
        bootstrap=dict(requested=resamples,valid=resamples-skipped,single_class_discarded=skipped,seed=seed,
            unit='patient; retain all ECGs together',method='paired percentile 95% intervals; mean of seed metrics',
            limitation='Conditional on saved predictions; excludes retraining uncertainty'))
