"""Run the frozen Lead-I coverage check and conditional logistic feasibility study."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import multiprocessing
from pathlib import Path
import time
import traceback
import joblib
import numpy as np

from experiments.acs_omi_vmd_wst_vqc import eyeball_common as C
from experiments.acs_omi_vmd_wst_vqc import lead_i_study as S
from experiments.acs_omi_vmd_wst_vqc.loader import ACSDataset
from experiments.acs_omi_vmd_wst_vqc.scripts import eyeball_pilot_assessment as A
from experiments.acs_omi_vmd_wst_vqc.scripts.eyeball_pilot import perturbations
from experiments.acs_omi_vmd_wst_vqc.reporting import binary_metrics


def prepare():
    spec=C.read(S.SPEC)
    parent,parent_sha=A.check_manifest(S.PARENT)
    A.compatible(parent['protocol'],spec['extractor'])
    with (S.PARENT/'selection.csv').open(newline='') as stream:
        prior=list(C.csv.DictReader(stream))
    source=C.split_rows()
    rows=S.select(source,{r['patient_id'] for r in prior},spec['cohort'])
    extra=[Path(__file__),Path(S.__file__),Path(A.__file__),S.SPEC,
        C.BASE/'tests/test_lead_i_study.py',C.BASE/'tests/test_eyeball_assessment.py',C.BASE/'reporting.py']
    code={**C.source_hashes(),**{str(p.relative_to(C.ROOT)):C.sha(p) for p in extra}}
    value=dict(protocol=spec,environment={**C.environment(),'joblib':importlib.metadata.version('joblib')},
        code_sha256=code,parent_manifest_sha256=parent_sha,
        parent_completion_sha256=C.sha(S.PARENT/'completed.json'),
        parent_selection_sha256=C.sha(S.PARENT/'selection.csv'),
        source_manifest_sha256=C.sha(C.SOURCE/'manifest.json'),splits_sha256=C.sha(C.SOURCE/'splits.csv'),
        cohort=rows)
    p=S.OUT/'manifest.json'
    if p.exists():
        if C.read(p)!=value or C.sha(p)!=(S.OUT/'manifest.sha256').read_text().strip():
            raise ValueError('Changed study; create a new version')
    else:
        if not C.verified(S.PARENT,parent_sha):raise ValueError('Parent assessment incomplete')
        C.write(p,value);C.put(S.OUT/'manifest.sha256',(C.sha(p)+'\n').encode())
        for name in code:C.put(S.OUT/'source'/name,(C.ROOT/name).read_bytes())
        C.csv_write(S.OUT/'cohort.csv',rows)
        C.write(S.OUT/'exclusions.json',dict(reason='Prior pilot patients excluded prospectively',
            patient_ids=sorted({r['patient_id'] for r in prior}),all_other_exclusions='Original frozen eligible-cohort policy'))
        C.event(S.OUT,'prepared',patients=len(rows),context_sha=C.sha(p))
    if not (S.OUT/'cohort.csv').exists():raise ValueError('Missing cohort ledger')
    with (S.OUT/'cohort.csv').open(newline='') as stream:
        if list(C.csv.DictReader(stream))!=rows:raise ValueError('Changed cohort ledger')
    return spec,rows,C.sha(p)


def leaf(variant,rid):return S.OUT/'extractions'/variant/rid


def verify_terminal(folder,digest,identity):
    value=C.read(folder/'terminal.json')
    if value['signature']['context_sha']!=digest or value['signature']['identity']!=identity:
        raise ValueError('Changed terminal identity')
    for name,sha in value['files'].items():
        if C.sha(folder/name)!=sha:raise ValueError('Changed terminal artifact')
    if value['outcome']=='success' and not C.verified(folder,digest):raise ValueError('Missing successful output')
    return value


def identity(row,variant,seed=None):
    result=dict(record_id=row['record_id'],patient_id=row['patient_id'],lead='I',
                partition=row['study_partition'],variant=variant)
    if seed is not None:result['seed']=seed
    return result


def extract_stage(ds,pool,rows,variant,stage,spec,digest,seed_map):
    folder=S.OUT/'stages'/stage
    if C.verified(folder,digest):
        for row in rows:
            verify_terminal(leaf(variant,row['record_id']),digest,
                identity(row,variant,None if variant=='clean' else seed_map[row['record_id']]))
        print(f'{stage}: verified saved stage',flush=True)
        return C.read(folder/'outcomes.json')
    C.event(S.OUT,'stage_started',stage=stage,attempts=len(rows))
    pending={};outcomes={};started=time.perf_counter()
    for row in rows:
        rid=row['record_id'];dest=leaf(variant,rid)
        seed=None if variant=='clean' else seed_map[rid]
        ident=identity(row,variant,seed)
        if (dest/'terminal.json').exists():
            outcomes[rid]=verify_terminal(dest,digest,ident);continue
        record=ds.load_record(rid)
        if (record.info.patient_id!=row['patient_id'] or record.info.split!='train'
                or record.info.label!=int(row['label']) or record.decision.waveform_sha256!=row['waveform_sha256']):
            raise ValueError('Changed input identity')
        x=record.lead('I').copy()
        if variant!='clean':x=perturbations(x,500,seed)[variant]
        future=pool.submit(A.task,dest,x,spec['extractor'],digest,ident,False)
        pending[future]=rid
    for future in as_completed(pending):
        rid=pending[future];outcomes[rid]=future.result()
        failures=sum(r['outcome']!='success' for r in outcomes.values())
        C.write(S.OUT/'status.json',dict(status='running',stage=stage,completed=len(outcomes),
            total=len(rows),numerical_failures=failures,time_utc=C.now()))
        if len(outcomes)%32==0 or len(outcomes)==len(rows) or outcomes[rid]['outcome']!='success':
            print(f'{stage}: {len(outcomes)}/{len(rows)}; failures={failures}',flush=True)
    if set(outcomes)!={r['record_id'] for r in rows}:raise ValueError('Incomplete stage')
    C.write(folder/'outcomes.json',outcomes)
    valid=[outcomes[r['record_id']]['outcome']=='success' for r in rows]
    result=dict(coverage=S.coverage(rows,valid),seconds_this_session=time.perf_counter()-started,
        errors={r['record_id']:outcomes[r['record_id']].get('error') for r in rows if outcomes[r['record_id']]['outcome']!='success'})
    C.write(folder/'summary.json',result);C.complete(folder,['outcomes.json','summary.json'],digest)
    C.event(S.OUT,'stage_complete',stage=stage,**result)
    return outcomes


def matrix(rows,variant):
    X=np.full((len(rows),12),np.nan);valid=np.zeros(len(rows),dtype=bool)
    for i,row in enumerate(rows):
        dest=leaf(variant,row['record_id'])
        if C.read(dest/'terminal.json')['outcome']=='success':
            X[i]=C.arrays(dest/'features.npz')['features'];valid[i]=True
    if not np.isfinite(X[valid]).all():raise ValueError('Invalid accepted features')
    return X,valid


def export_report(digest):
    completion=C.read(S.OUT/'completed.json')
    export_names=['summary.json','report.md','cohort.csv','exclusions.json','manifest.json']
    export_names += [name for name in ['analysis.json','predictions.csv','bootstrap_draws.npz','models/fit_details.json'] if (S.OUT/name).exists()]
    for name in export_names:C.put(S.REPORT/name,(S.OUT/name).read_bytes())
    C.write(S.REPORT/'artifact_check.json',dict(time_utc=C.now(),run_root=str(S.OUT.relative_to(C.ROOT)),
        completed_sha256=C.sha(S.OUT/'completed.json'),entries=len(completion['files']),exported=export_names))
    C.complete(S.REPORT,export_names+['artifact_check.json'],digest)


def finish(summary,report_lines,digest):
    C.write(S.OUT/'summary.json',summary)
    C.put(S.OUT/'report.md',('\n'.join(report_lines)+'\n').encode())
    names=[str(p.relative_to(S.OUT)) for p in sorted(S.OUT.rglob('*')) if p.is_file()
        and p.relative_to(S.OUT).as_posix() not in ('completed.json','status.json','attempts.jsonl','.writer.lock')]
    C.complete(S.OUT,names,digest)
    C.write(S.OUT/'status.json',summary)
    C.event(S.OUT,'complete',decision=summary['decision'],models_trained=summary['models_trained'])
    export_report(digest)
    print(C.json.dumps(summary,indent=2),flush=True)


def negative(summary,digest):
    lines=['# Lead-I feasibility study — extraction gate not passed','',
        summary['decision'], '',
        'The declared study ended before evaluation extraction and classifier fitting. No accuracy, AP or predictions were produced. '
        'Every attempted fit ECG and numerical failure remains saved; no patient was replaced and no cap was increased.', '',
        f"Engineering coverage: {summary['engineering_coverage']['successful']}/{summary['engineering_coverage']['attempted']}; "
        'the prospective requirement was at least 99%. This is an engineering threshold, not a clinical standard.', '',
        'The extractor is the unchanged 64k-bound, four-IMF, 10-second Lead-I adaptation. '
        'All patients are from the original fit partition, with the earlier 32 pilot patients excluded. '
        'The original validation and official test patients remain reserved.', '',
        'See [summary](summary.json), [cohort](cohort.csv) and [frozen protocol and source manifest](manifest.json).']
    finish(summary,lines,digest)


def run():
    started=time.perf_counter()
    with C.lock(S.OUT):
        spec,rows,digest=prepare()
        if C.verified(S.OUT,digest):
            if not C.verified(S.REPORT,digest):
                # A crash during export can be repaired without extraction/fitting.
                export_report(digest)
            print('Verified completed Lead-I study; no computation repeated',flush=True);return
        fit=[r for r in rows if r['study_partition']=='model_fit']
        evaluation=[r for r in rows if r['study_partition']=='model_evaluation']
        engineering=[r for r in rows if r['engineering']=='1']
        seed_map={r['record_id']:spec['perturbations']['seed_base']+i for i,r in enumerate(evaluation)}
        C.write(S.OUT/'status.json',dict(status='running',stage='verify_archives',time_utc=C.now()))
        C.event(S.OUT,'started',context_sha=digest)
        summary=dict(status='complete',study=spec['id'],models_trained=0,
            original_validation_processed=False,official_test_processed=False)
        try:
            with ACSDataset(target='OMI') as ds, ProcessPoolExecutor(max_workers=spec['execution']['workers'],
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                outcomes=extract_stage(ds,pool,engineering,'clean','engineering',spec,digest,seed_map)
                valid=[outcomes[r['record_id']]['outcome']=='success' for r in engineering]
                summary['engineering_coverage']=S.coverage(engineering,valid)
                if np.mean(valid)<spec['gate']['minimum_engineering_coverage']:
                    summary.update(decision='Stop: larger Lead-I engineering coverage is below the declared 99% threshold.',
                        seconds_this_session=time.perf_counter()-started)
                    negative(summary,digest);return
                extract_stage(ds,pool,fit,'clean','model_fit_clean',spec,digest,seed_map)
                X,fit_valid=matrix(fit,'clean');y=np.array([int(r['label']) for r in fit])
                summary['model_fit_coverage']=S.coverage(fit,fit_valid)
                y_eval=np.array([int(r['label']) for r in evaluation])
                enough=(fit_valid.mean()>=spec['gate']['minimum_model_fit_coverage']
                    and int(((y==1)&fit_valid).sum())>=spec['gate']['minimum_training_positives']
                    and int(((y==0)&fit_valid).sum())>=spec['gate']['minimum_training_negatives']
                    and int((y_eval==1).sum())>=spec['gate']['minimum_evaluation_positives']
                    and int((y_eval==0).sum())>=spec['gate']['minimum_evaluation_negatives'])
                if not enough:
                    summary.update(decision='Stop: model-fit coverage or declared class-count requirements were not met.',
                        seconds_this_session=time.perf_counter()-started)
                    negative(summary,digest);return
                fallback=float(y.mean());model_dir=S.OUT/'models'
                if C.verified(model_dir,digest):
                    models=joblib.load(model_dir/'models.joblib');fit_details=C.read(model_dir/'fit_details.json')
                else:
                    models,fit_details=S.fit_models(X,y,fit_valid,spec['classifiers']['shared'])
                    S.save_models(model_dir/'models.joblib',models)
                    reloaded=joblib.load(model_dir/'models.joblib')
                    expected=S.predict(models,X,fit_valid,fallback);actual=S.predict(reloaded,X,fit_valid,fallback)
                    for name in expected:np.testing.assert_array_equal(expected[name],actual[name])
                    C.write(model_dir/'fit_details.json',dict(models=fit_details,selected_fit_prevalence=fallback,
                        omitted_fit_record_ids=[r['record_id'] for r,v in zip(fit,fit_valid) if not v],exact_reload_match=True))
                    C.complete(model_dir,['models.joblib','fit_details.json'],digest)
                summary['models_trained']=2
                scores={};coverages={};validities={};prediction_rows=[];covered_metrics={}
                for variant in ['clean']+spec['perturbations']['evaluation_variants']:
                    extract_stage(ds,pool,evaluation,variant,'evaluation_'+variant,spec,digest,seed_map)
                    X_eval,valid_eval=matrix(evaluation,variant)
                    variant_scores=S.predict(models,X_eval,valid_eval,fallback)
                    coverages[variant]=S.coverage(evaluation,valid_eval);validities[variant]=valid_eval
                    for name,values in variant_scores.items():
                        key=name+'__'+variant;scores[key]=values
                        covered_metrics[key]=binary_metrics(y_eval[valid_eval],values[valid_eval]) if len(np.unique(y_eval[valid_eval]))==2 else None
                        for i,row in enumerate(evaluation):
                            prediction_rows.append(dict(record_id=row['record_id'],patient_id=row['patient_id'],
                                label=int(row['label']),study_partition='model_evaluation',variant=variant,model=name,
                                probability=float(values[i]),predicted_label=int(values[i]>=.5),
                                extraction_succeeded=bool(valid_eval[i]),fallback_used=bool(not valid_eval[i] and name!='constant_prior'),
                                model_seed=0,noise_seed='' if variant=='clean' else seed_map[row['record_id']]))
                C.csv_write(S.OUT/'predictions.csv',prediction_rows)
            pairs={'geometry_clean':('full_geometry__clean','without_geometry__clean'),
                'full_vs_prior_clean':('full_geometry__clean','constant_prior__clean')}
            stability={}
            for model in ('full_geometry','without_geometry'):
                clean=scores[model+'__clean']
                for variant in spec['perturbations']['evaluation_variants']:
                    key=model+'__'+variant;pairs[key+'_minus_clean']=(key,model+'__clean')
                    perturbed=scores[key];both=validities['clean']&validities[variant]
                    stability[key]=dict(mean_absolute_score_change=float(np.mean(abs(perturbed-clean))),
                        p95_absolute_score_change=float(np.quantile(abs(perturbed-clean),.95)),
                        threshold_flip_count=int(((perturbed>=.5)!=(clean>=.5)).sum()),
                        both_valid_patients=int(both.sum()),
                        both_valid_mean_absolute_score_change=float(np.mean(abs(perturbed[both]-clean[both]))) if both.any() else None)
            C.write(S.OUT/'status.json',dict(status='running',stage='paired_bootstrap',time_utc=C.now()))
            analysis,draws=S.resample_metrics(y_eval,np.array([r['patient_id'] for r in evaluation]),scores,pairs,
                spec['evaluation']['bootstrap_draws'],spec['evaluation']['bootstrap_seed'])
            analysis.update(coverage=coverages,covered_only_metrics=covered_metrics,prediction_stability=stability,
                fallback_probability=fallback,selected_fit_class_counts={str(k):int((y==k).sum()) for k in (0,1)},
                selected_evaluation_class_counts={str(k):int((y_eval==k).sum()) for k in (0,1)})
            C.write(S.OUT/'analysis.json',analysis);C.npz(S.OUT/'bootstrap_draws.npz',**draws)
            summary.update(decision='Completed the declared internal Lead-I logistic feasibility comparison.',
                evaluation_patients=len(evaluation),evaluation_coverage=coverages,
                metrics=analysis['metrics'],seconds_this_session=time.perf_counter()-started)
            lines=['# Lead-I logistic feasibility results','',summary['decision'],'',
                'This is a new internal holdout inside the original fit partition. The prior 32 pilot patients were excluded. '
                'No original-validation or official-test ECG was evaluated; these scores cannot directly rank against the earlier LFCC/Swin validation scores.', '',
                f"Selected model-fit patients: {len(fit)}; internal evaluation patients: {len(evaluation)} "
                f"({int(y_eval.sum())} OMI-positive). Both logistic fits use class weights, C=1, training-only StandardScaler, seed 0 and threshold 0.5. No tuning or oversampling.",'',
                '| Model / input | AP | Accuracy | Sensitivity | Specificity |','|---|---:|---:|---:|---:|']
            for name,m in analysis['metrics'].items():
                lines.append(f"| {name} | {m['average_precision']:.4f} | {m['accuracy']:.2%} | {m['sensitivity']:.2%} | {m['specificity']:.2%} |")
            contrast=analysis['comparisons']['geometry_clean']
            lines += ['',f"Primary full-minus-no-geometry AP difference: {contrast['point']['average_precision']:+.4f}; "
                f"95% paired patient-bootstrap interval {contrast['intervals']['average_precision']}.", '',
                'All selected evaluation patients are included above. Failed extraction receives the explicitly flagged selected-training-prevalence fallback; '
                'it is not an accepted feature vector or an ECG-derived classifier score. The analysis also reports covered-only metrics and coverage by class. '
                'Invalid training records are listed as omitted from fitting; no replacement patients were sampled.', '',
                '| Evaluation input | Extraction successes |','|---|---:|']
            lines += [f"| {name} | {v['successful']}/{v['attempted']} |" for name,v in coverages.items()]
            lines += ['', 'The unchanged extractor uses Lead I, native 500 Hz physical-mV input, four PyEMD 1.6.4 IMFs, a 64k ceiling, '
                '0.25-second trimming and the previously declared 12 descriptors. Noise seeds and every prediction are in the CSV. '
                'Models are held fixed for the 30 dB, 20 dB and 0.1-second-per-end cropping conditions.', '',
                'Intervals use 2,000 paired patient draws (seed 20261009), condition on fitted models and this small selected cohort, and exclude retraining uncertainty. '
                'They are pointwise and exploratory. This study does not establish clinical utility, external validity or quantum advantage. '
                'No VQC or Swin model was trained.', '',
                '[All predictions](predictions.csv) · [Metrics, uncertainty, coverage and stability](analysis.json) · '
                '[Cohort](cohort.csv) · [Fit diagnostics](models/fit_details.json) · [Protocol/source manifest](manifest.json)']
            finish(summary,lines,digest)
        except BaseException:
            failure=dict(status='failed',time_utc=C.now(),traceback=traceback.format_exc())
            C.write(S.OUT/'status.json',failure);C.event(S.OUT,'failed',traceback=failure['traceback']);raise


if __name__=='__main__':run()
