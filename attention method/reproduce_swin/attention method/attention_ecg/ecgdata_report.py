"""Patient-cluster reporting from saved predictions only; never fits a model."""
from pathlib import Path
import numpy as np

from .ecgdata import REPO, DEFAULT_CACHE, safe_output, verify_cache, read_json, file_hash, atomic_json
from .ecgdata_training import DEFAULT_RUN, verify_trial


def confusion(y, pred, weights=None):
    return np.bincount(3*np.asarray(y,dtype=int)+np.asarray(pred,dtype=int),
                       weights=weights, minlength=9).reshape(3,3)


def scores(cm):
    cm = np.asarray(cm,dtype=float)
    tp, true, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    f1 = np.divide(2*tp,true+predicted,out=np.zeros(3),where=(true+predicted)>0)
    recall = np.divide(tp,true,out=np.zeros(3),where=true>0)
    precision = np.divide(tp,predicted,out=np.zeros(3),where=predicted>0)
    tn = cm.sum()-true-predicted+tp
    specificity = np.divide(tn,cm.sum()-true,out=np.zeros(3),where=(cm.sum()-true)>0)
    return dict(accuracy=float(tp.sum()/cm.sum()), macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()), precision=precision.tolist(),
                sensitivity=recall.tolist(), specificity=specificity.tolist(), f1=f1.tolist(),
                confusion_matrix=cm.tolist())


def patient_confusions(y, pred, patients):
    window, vote = [], []
    for patient in np.unique(patients):
        idx = patients == patient
        labels = np.unique(y[idx])
        if len(labels)!=1:
            raise ValueError('Patient label inconsistent')
        prediction = np.bincount(pred[idx],minlength=3).argmax()
        window.append(confusion(y[idx],pred[idx]))
        vote.append(confusion(labels,np.array([prediction])))
    return np.array(window), np.array(vote)


def metric_vector(window_cm, patient_cm):
    w, p = scores(window_cm), scores(patient_cm)
    return np.array([w['macro_f1'],w['accuracy'],p['macro_f1'],p['accuracy']])


def bootstrap_comparison(y, patients, predictions, repetitions, seed):
    """Same patient multiplicities for every arm/seed; average seeds inside draws."""
    grouped = {arm:[patient_confusions(y,pred,patients) for pred in seeds]
               for arm,seeds in predictions.items()}
    n = len(np.unique(patients))
    rng = np.random.default_rng(seed)
    draws = {arm:[] for arm in predictions}
    rejected=0
    for _ in range(repetitions):
        mult = np.bincount(rng.integers(n,size=n),minlength=n)
        reference = next(iter(grouped.values()))[0][0]
        if np.any((reference*mult[:,None,None]).sum((0,2)) == 0):
            rejected+=1
            continue
        for arm,items in grouped.items():
            values=[metric_vector((w*mult[:,None,None]).sum(0),(p*mult[:,None,None]).sum(0)) for w,p in items]
            draws[arm].append(np.mean(values,axis=0))
    if not draws or not len(next(iter(draws.values()))):
        raise ValueError('No valid bootstrap draws')
    keys=['window_macro_f1','window_accuracy','patient_vote_macro_f1','patient_vote_accuracy']
    intervals={}
    for arm,values in draws.items():
        bounds=np.quantile(values,[.025,.975],axis=0)
        intervals[arm]={key:bounds[:,i].tolist() for i,key in enumerate(keys)}
    differences={}
    for arm in predictions:
        if arm=='swin':
            continue
        bounds=np.quantile(np.array(draws['swin'])-np.array(draws[arm]),[.025,.975],axis=0)
        differences[f'swin_minus_{arm}']={key:bounds[:,i].tolist() for i,key in enumerate(keys)}
    return dict(repetitions=repetitions,seed=seed,rejected_single_class_draws=rejected,
                intervals_95=intervals,paired_difference_intervals_95=differences)


def collect_predictions(run,data,protocol):
    predictions={}
    for arm,seeds in [('swin',protocol['training']['seeds']),('logistic',[None])]:
        full=[]
        for seed in seeds:
            pred=np.full(len(data['y']),-1,dtype=int)
            coverage=np.zeros(len(pred),dtype=int)
            for fold in range(5):
                name=f'swin_seed{seed}_fold{fold}' if arm=='swin' else f'logistic_fold{fold}'
                folder=run/name
                if not verify_trial(folder):
                    raise ValueError(f'Incomplete trial: {name}')
                fit=read_json(folder/'fit.json')
                expected=np.flatnonzero(data['fold']==fold)
                if fit['train_indices']!=np.flatnonzero(data['fold']!=fold).tolist() or fit['test_indices']!=expected.tolist():
                    raise ValueError('Saved training/test boundaries changed')
                with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                    probs=z['probabilities']
                    if (not np.array_equal(z['indices'],expected) or not np.array_equal(z['y'],data['y'][expected])
                            or not np.array_equal(z['patients'],data['patients'][expected]) or probs.shape!=(len(expected),3)
                            or not np.isfinite(probs).all() or np.any(probs<0) or not np.allclose(probs.sum(1),1)
                            or not np.array_equal(z['pred'],probs.argmax(1))):
                        raise ValueError('Prediction alignment/values invalid')
                    pred[expected]=z['pred']
                    coverage[expected]+=1
            if not np.all(coverage==1):
                raise ValueError('Predictions must cover each window exactly once')
            full.append(pred)
        predictions[arm]=full
    return predictions


def make_report(cache=DEFAULT_CACHE, run=DEFAULT_RUN):
    run=safe_output(run,'results')
    manifest,data=verify_cache(cache)
    protocol=manifest['protocol']
    if not verify_trial(run):
        raise ValueError('All 20 planned fits must finish before reporting')
    training=read_json(run/'manifest.json')
    if training['cache_completed_sha256']!=file_hash(Path(cache)/'completed.json'):
        raise ValueError('Report uses another feature cache')
    predictions=collect_predictions(run,data,protocol)
    reference=REPO/protocol['reference']['directory']/'predictions.npz'
    with np.load(reference,allow_pickle=False) as z:
        for key,local in [('__y','labels'),('__groups','patients'),('__sources','sources'),('__row_ids','row_ids'),('__fold','fold')]:
            if not np.array_equal(z[key],data[local]):
                raise ValueError('Original VQC predictions do not align')
        for name,key in [('original_vmd_vqc','VMD descriptors'),('original_wst_vqc','WST log')]:
            predictions[name]=[np.array([protocol['classes'].index(str(label)) for label in z[f'{key} + VQC seed={seed}']])
                               for seed in protocol['training']['seeds']]
    results={}
    for arm,items in predictions.items():
        per_seed=[]
        for pred in items:
            windows,patients=patient_confusions(data['y'],pred,data['patients'])
            per_seed.append(dict(window=scores(windows.sum(0)),patient_vote=scores(patients.sum(0))))
        results[arm]=dict(per_seed=per_seed, mean={scope:{metric:float(np.mean([r[scope][metric] for r in per_seed]))
                    for metric in ('accuracy','macro_f1','balanced_accuracy')} for scope in ('window','patient_vote')})
    e=protocol['evaluation']
    report=dict(results=results,classes=protocol['classes'],seed_order=protocol['training']['seeds'],
                logistic_is_deterministic=True,
                bootstrap=bootstrap_comparison(data['y'],data['patients'],predictions,e['bootstrap_resamples'],e['bootstrap_seed']),
                uncertainty_scope=e['interval_scope'],original_predictions_sha256=file_hash(reference),
                report_source_sha256=file_hash(__file__),training_manifest_sha256=file_hash(run/'manifest.json'))
    if (run/'report.json').exists() and read_json(run/'report.json')!=report:
        raise ValueError('A different report already exists; preserve it and investigate')
    atomic_json(run/'report.json',report)
    return report
