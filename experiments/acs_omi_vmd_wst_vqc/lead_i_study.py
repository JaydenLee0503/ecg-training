"""Predeclared sampling, fit-only preprocessing, and failure-aware predictions."""
import io
import warnings
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc.reporting import binary_metrics, METRICS

SPEC = C.BASE/'protocols/eyeball_lead_i_feasibility_v1.json'
OUT = C.BASE/'results/eyeball_lead_i_feasibility_v1'
PARENT = C.BASE/'results/eyeball_omi_pilot_retry64k_v1'
REPORT = C.BASE/'reports/eyeball_lead_i_feasibility_v1'


def rank(identifier, value):
    return C.hashlib.sha256(f'{identifier}|{value}'.encode()).hexdigest(), value


def select(rows, excluded, settings):
    eligible = [r for r in rows if r['partition']=='fit' and r['patient_id'] not in excluded]
    chosen = C.selection(eligible, settings['patients'], settings['selection_id'])
    if len(chosen)!=settings['patients']:
        raise ValueError('Insufficient eligible patients')
    ordered = sorted(chosen,key=lambda r:rank(settings['split_id'],r['patient_id']))
    evaluation = {r['patient_id'] for r in ordered[:settings['model_evaluation_patients']]}
    fit = [r for r in chosen if r['patient_id'] not in evaluation]
    if len(fit)!=settings['model_fit_patients']:
        raise ValueError('Inconsistent declared split counts')
    engineering = {r['patient_id'] for r in sorted(fit,key=lambda r:rank(settings['engineering_id'],r['patient_id']))[:settings['engineering_patients']]}
    return sorted([dict(r,study_partition='model_evaluation' if r['patient_id'] in evaluation else 'model_fit',
                        engineering='1' if r['patient_id'] in engineering else '0') for r in chosen],key=lambda r:r['record_id'])


def coverage(rows, valid):
    y = np.array([int(r['label']) for r in rows]); valid=np.asarray(valid,dtype=bool)
    if len(valid)!=len(y): raise ValueError('Coverage length mismatch')
    return dict(attempted=len(y),successful=int(valid.sum()),failed=int((~valid).sum()),
        fraction=float(valid.mean()),by_class={str(label):dict(attempted=int((y==label).sum()),
            successful=int((valid & (y==label)).sum()),failed=int((~valid & (y==label)).sum())) for label in (0,1)})


def fit_models(X, y, valid, settings):
    X=np.asarray(X);y=np.asarray(y);valid=np.asarray(valid,dtype=bool)
    if X.shape!=(len(y),12) or valid.shape!=y.shape or not np.isfinite(X[valid]).all():
        raise ValueError('Invalid training matrix')
    models={}; details={}
    # Validation arrays are deliberately not accepted by this function.
    for name, width in [('full_geometry',12),('without_geometry',10)]:
        model=make_pipeline(StandardScaler(),LogisticRegression(**{k:settings[k] for k in
            ('C','solver','max_iter','class_weight','random_state')}))
        started=C.time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(X[valid,:width],y[valid])
        models[name]=model
        details[name]=dict(seconds=C.time.perf_counter()-started,training_rows=int(valid.sum()),
            warnings=[str(w.message) for w in caught],iterations=model[-1].n_iter_.tolist(),features=width)
    return models, details


def predict(models, X, valid, fallback):
    X=np.asarray(X);valid=np.asarray(valid,dtype=bool)
    if X.shape!=(len(valid),12) or not np.isfinite(X[valid]).all() or not 0<=fallback<=1:
        raise ValueError('Invalid evaluation inputs')
    scores={'constant_prior':np.full(len(valid),fallback)}
    for name, width in [('full_geometry',12),('without_geometry',10)]:
        values=np.full(len(valid),fallback)
        if valid.any():values[valid]=models[name].predict_proba(X[valid,:width])[:,1]
        scores[name]=values
    return scores


def save_models(path, models):
    stream=io.BytesIO();joblib.dump(models,stream);C.put(path,stream.getvalue())


def resample_metrics(y, patients, scores, pairs, count, seed):
    """Keep the same patient draw for every clean/perturbed arm; save all draws."""
    y=np.asarray(y);patients=np.asarray(patients)
    if len(set(patients))!=len(patients):raise ValueError('Study requires one ECG per patient')
    point={name:binary_metrics(y,score) for name,score in scores.items()}
    rng=np.random.default_rng(seed);draws={name:[] for name in scores};indices=[];skipped=0
    for i in range(count):
        draw=rng.integers(0,len(y),len(y))
        if len(np.unique(y[draw]))<2:skipped+=1;continue
        indices.append(draw)
        for name, score in scores.items():
            metrics=binary_metrics(y[draw],score[draw])
            draws[name].append([metrics[k] for k in METRICS])
        if (i+1)%200==0: print(f'Bootstrap {i+1}/{count}',flush=True)
    if not indices:raise ValueError('No valid bootstrap draws')
    draws={name:np.asarray(values) for name,values in draws.items()}
    def intervals(a):
        low,high=np.quantile(a,[.025,.975],axis=0)
        return {k:[float(x),float(z)] for k,x,z in zip(METRICS,low,high)}
    differences={name:dict(left=a,right=b,point={k:point[a][k]-point[b][k] for k in METRICS},
        intervals=intervals(draws[a]-draws[b])) for name,(a,b) in pairs.items()}
    return dict(metrics=point,intervals={name:intervals(a) for name,a in draws.items()},
        comparisons=differences,bootstrap=dict(requested=count,valid=len(indices),discarded_single_class=skipped,seed=seed)),dict(**draws,patient_indices=np.asarray(indices),metric_names=np.array(METRICS))
