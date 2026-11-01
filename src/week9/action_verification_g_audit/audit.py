"""Independent saved-output audit. Never trains a learner or reruns the OOF study."""
import argparse
import ast
import csv
import json
from pathlib import Path
from collections import defaultdict
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib
from ..provenance import ROOT,branch_guard,digest,file_hash
from ..caching import read_jsonl,atomic_json
from ..action_verification_g import protocol,features,controller

VALID_STATUSES=('VALID','VALID WITH NON-MATERIAL FIXES')


def check(condition,message):
    if not condition:raise AssertionError(message)


def csv_rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def number(value):return float(value)


def close(a,b,message):check(np.isclose(float(a),float(b),atol=1e-12,rtol=1e-12),message)


def pick_threshold(rows,tolerance=.003):
    """Independent literal reconstruction of the frozen threshold rule."""
    best=max(float(r['f1']) for r in rows)
    eligible=[r for r in rows if float(r['f1'])>=best-tolerance-1e-12]
    selected=min(eligible,key=lambda r:(float(r['hurt']),float(r['acceptance_rate']),
        float(r['total_query_prompt_tokens']),abs(float(r['threshold'])-.5),-float(r['threshold'])))
    return selected,best,eligible


def bootstrap(left,right,n=10000,seed=42):
    """Resample complete paper clusters, weighted by their included query counts."""
    check(set(left)==set(right),'Bootstrap requires identical query keys')
    clusters=defaultdict(list)
    for q,r in left.items():
        check(r['paper_id']==right[q]['paper_id'],'Bootstrap cross-paper query')
        clusters[r['paper_id']].append(float(r['f1'])-float(right[q]['f1']))
    order=sorted(clusters)
    sums=np.array([sum(clusters[p]) for p in order]);sizes=np.array([len(clusters[p]) for p in order])
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,n,500):
        draws=rng.integers(0,len(order),size=(min(500,n-start),len(order)))
        samples.extend((sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)).tolist())
    return dict(point_estimate=float(sums.sum()/sizes.sum()),ci95_low=float(np.quantile(samples,.025)),
                ci95_high=float(np.quantile(samples,.975)),cluster='paper_id',papers=len(order),resamples=n,seed=seed)


def load_case():
    branch_guard();cfg=protocol.config(ROOT/'configs/week9/w9g_action_verification.yaml')
    bundle=protocol.inputs(cfg);ident=protocol.identity(cfg,bundle)
    root=protocol.resolve(cfg['output_root']);oof=root/'controller_fit_oof'
    p=json.loads((oof/'protocol.json').read_text());manifest=json.loads((oof/'manifest.json').read_text())
    check(manifest['complete'] and manifest['split']=='fit','Completed FIT OOF required')
    check(manifest['protocol_hash']==digest(p),'Controller protocol hash mismatch')
    protocol.verify_hashes(manifest['artifact_hashes']);protocol.verify_hashes(p['sources'])
    check(p['config']==cfg and p['preflight_protocol_hash']==digest(ident),'Frozen training/preflight config changed')
    check(p['sklearn']==version('scikit-learn'),'Audit sklearn differs from saved training version')
    x,meta=features.prepare(bundle,cfg)
    check(len(meta)==1746 and meta.paper_id.nunique()==656,'FIT query/paper counts')
    check(len(x.columns)==32 and list(x.columns)==p['features'],'Frozen 32-feature order changed')
    saved=pd.read_csv(root/'scaffold/inference_features.csv',dtype={'question_id':str,'paper_id':str})
    check(saved[['question_id','paper_id']].equals(meta),'Saved feature metadata/order differs')
    check(np.allclose(saved[list(x.columns)].to_numpy(),x.to_numpy(),equal_nan=True,atol=1e-12,rtol=1e-12),
        'Reconstructed feature matrix differs from saved scaffold')
    protected=dict(bundle['input_hashes'])
    for folder in (root/'preflight',root/'scaffold',oof):
        protected.update({str(f.resolve()):file_hash(f) for f in folder.rglob('*') if f.is_file()})
    for path in (ROOT/'configs/week9/w9g_action_verification.yaml',ROOT/'notebooks/week9/week9g_action_verification.ipynb'):
        protected[str(path)]=file_hash(path)
    protected.update({str(ROOT/k):v for k,v in p['sources'].items()})
    return dict(cfg=cfg,bundle=bundle,ident=ident,root=root,oof=oof,p=p,manifest=manifest,x=x,meta=meta,protected=protected)


def audit_lineage(case):
    bundle=case['bundle'];g=bundle['groups'];fp=json.loads((protocol.resolve(case['cfg']['f_root'])/'protocol.json').read_text())
    paths=[p for p in fp['input_hashes'] if p.endswith('/w9d_representation.jsonl')]
    check(len(paths)==1,'Original D0 frozen source not uniquely identified')
    canonical={r['question_id']:r for r in read_jsonl(paths[0]) if r['policy']=='D0'}
    records=[]
    for q in sorted(g['D0']):
        base=g['D0'][q];fresh=g['F0'][q];src=bundle['e3'][q];old=canonical[q]
        check(old['paper_id']==base['paper_id'] and digest(old['prediction'])==digest(base['prediction']), 'Canonical D0 answer hash mismatch')
        check(fresh['model']==fp['model'] and fresh['decoding']==dict(do_sample=False,max_new_tokens=128),'F0 generator/decoding differs')
        check(fresh['final_prompt_hash']==src['final_prompt_hash'] and fresh['tokenizer']==src['tokenizer'],'F0 prompt/tokenizer mismatch')
        check(fresh['answer_output_hash']==digest(fresh['prediction'])==src['answer_output_hash'],'F0 answer hash mismatch')
        check(fresh['selected_chunk_ids']==src['selected_chunk_ids'] and fresh['source_chunk_hashes']==src['source_chunk_hashes'],'Source chunks differ')
        check(fresh['evidence_hash']==digest(fresh['evidence_text'])==src['evidence_hash'],'E3 evidence hash mismatch')
        check(fresh['selected_sentence_ids']==src['selected_sentence_ids'],'E3 sentence IDs mismatch')
        delta=fresh['f1']-base['f1']
        label=1 if delta>.01 else 0
        records.append(dict(question_id=q,paper_id=base['paper_id'],d0_answer_hash=digest(base['prediction']),
            f0_answer_hash=digest(fresh['prediction']),evidence_hash=fresh['evidence_hash'],
            sentence_ids_hash=digest(fresh['selected_sentence_ids']),chunk_ids_hash=digest(fresh['selected_chunk_ids']),
            model_identity_hash=digest(fresh['model']),prompt_hash=fresh['final_prompt_hash'],
            delta_action=delta,accept_this_f0=label,state='HELP' if delta>.01 else 'HURT' if delta<-.01 else 'NEUTRAL'))
    saved=csv_rows(case['root']/'preflight/action_targets_fit_only.csv');by={r['question_id']:r for r in saved}
    check(len(by)==len(saved)==1746,'Target coverage/duplicates')
    for r in records:
        s=by[r['question_id']]
        close(s['delta_action'],r['delta_action'],'Saved action delta mismatch')
        check(int(s['accept_f0'])==r['accept_this_f0'] and s['action_state']==r['state'],'Wrong action-specific target')
    return records


def feature_audit(case):
    x=case['x'];by={name:group for group,names in features.GROUPS.items() for name in names}
    sources={
        'normalized_equal':'normalized tokens of y0 and specific y1','token_edit_distance':'Levenshtein(y0 tokens,y1 tokens)',
        'token_jaccard':'token-set overlap of y0/y1','added_token_count':'positive multiset y1-minus-y0',
        'removed_token_count':'positive multiset y0-minus-y1','initial_answer_length':'y0 token count',
        'candidate_answer_length':'y1 token count','answer_length_difference':'len(y1)-len(y0)',
        'answer_length_ratio':'len(y1)/max(len(y0),1)','changed_fraction':'edit distance/max answer length',
        'one_answer_substring':'normalized y0/y1 substring relationship',
        'initial_evidence_overlap':'lexical y0/E overlap','candidate_evidence_overlap':'lexical y1/E overlap',
        'evidence_overlap_delta':'candidate minus initial lexical overlap','added_tokens_in_evidence_fraction':'added y1 tokens present in E',
        'removed_tokens_absent_evidence_fraction':'removed y0 tokens absent from E',
        'initial_generated_tokens':'saved first-pass generation length','candidate_generated_tokens':'saved F0 generation length',
        'top_sentence_cosine':'first frozen E3 selected cosine','second_sentence_cosine':'second frozen E3 selected cosine',
        'third_sentence_cosine':'third frozen E3 selected cosine','mean_selected_cosine':'saved E3 selected cosine mean',
        'min_selected_cosine':'saved E3 selected cosine minimum','next_unselected_cosine':'saved E3 next-candidate cosine',
        'selected_vs_next_margin':'saved E3 minimum-selected minus next cosine','evidence_tokens':'frozen E3 evidence length',
        'mean_source_chunk_rank':'mean document-order selected chunk rank; no ID','selected_chunk_count':'number of selected source chunks',
        'query_tokens':'question token count','question_mark':'question terminal punctuation',
        'starts_wh_question':'fixed generic opening word set','starts_auxiliary_question':'fixed generic auxiliary opening word set'}
    tree=ast.parse(Path(features.__file__).read_text())
    check(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','fit_transform') for n in ast.walk(tree)),
        'Feature construction contains globally fitted preprocessing')
    records=[]
    for name in x:
        group=by[name]
        records.append(dict(feature=name,group=group,inference_available='yes, after F0 generation',source=sources[name],
            leakage_risk='Conditional frozen E3 selection used FIT globally; no per-query gold input' if group=='V3' else 'No outcome input; IDs/row position excluded',
            verdict='PASS',missing_count=int(x[name].isna().sum()),missing_fraction=float(x[name].isna().mean()),
            imputation='training-fold median + missing indicators; no global fit',
            observed_zero_count=int((x[name]==0).sum())))
    return records


def audit_models(case):
    result={};x=case['x'];saved=case['p']
    for name in ('logistic','hist_gradient_boosting'):
        pipeline=joblib.load(case['oof']/f'{name}.joblib')
        check(list(pipeline.feature_names_in_)==list(x.columns),'Serialized model feature schema differs')
        check(pipeline.n_features_in_==32,'Serialized model feature count')
        imputer=pipeline.steps[0][1]
        check(imputer.strategy=='median' and imputer.add_indicator,'Frozen imputer differs')
        check(np.allclose(imputer.statistics_,x.median().to_numpy(),equal_nan=True,atol=1e-12,rtol=1e-12),'Final imputer not consistent with full FIT medians')
        model=pipeline.steps[-1][1];params=model.get_params(deep=False)
        check(model.random_state==42 and list(model.classes_)==[0,1],'Saved final seed/classes differ')
        if name=='logistic':
            check(model.C==1 and model.class_weight=='balanced' and model.max_iter==1000,'Logistic hyperparameters changed')
            scaler=pipeline.steps[1][1]
            check(float(scaler.n_samples_seen_)==1746,'Existing final pipeline did not see all FIT samples')
            check(np.allclose(scaler.mean_,x.mean().to_numpy(),atol=1e-12,rtol=1e-12),'Final scaler not consistent with FIT means')
            check(np.allclose(scaler.var_,x.var(ddof=0).to_numpy(),atol=1e-12,rtol=1e-12),'Final scaler variance differs')
        else:
            check(model.max_iter==100 and model.max_leaf_nodes==7 and model.max_depth==3 and
                model.min_samples_leaf==20 and model.l2_regularization==1 and model.class_weight is None,'GBDT hyperparameters changed')
            check(model.n_iter_==100,'Saved GBDT iteration count differs')
        result[name]=dict(estimator_type=type(model).__name__,params=params,sklearn=saved['sklearn'],
            feature_count=32,imputation='median, add_indicator=True',
            standardized=name=='logistic',training_rows_evidence=1746 if name=='logistic' else 'Frozen source final fit uses full FIT',
            no_training_set_predictions_evaluated=True,model_file_hash=file_hash(case['oof']/f'{name}.joblib'))
    return result


def audit_folds(case):
    meta=case['meta'];folds,splits=controller.paper_folds(meta,5)
    saved=pd.read_csv(case['oof']/'fold_assignments.csv',dtype={'question_id':str,'paper_id':str})
    check(saved[['question_id','paper_id']].equals(meta),'Fold metadata differs')
    check(np.array_equal(saved.outer_fold.to_numpy(),folds),'Saved outer folds differ from frozen deterministic paper protocol')
    audits=json.loads((case['oof']/'fold_audit.json').read_text())
    check(len(audits)==40,'Expected 2 models x 5 outer x 4 inner fold audits')
    keyed={(a['model'],a['outer_fold'],a['inner_fold']):a for a in audits}
    check(len(keyed)==40,'Duplicate inner fold audit')
    records=[];training_classes=[]
    y=np.array([case['bundle']['groups']['F0'][q]['f1']-case['bundle']['groups']['D0'][q]['f1']>.01 for q in meta.question_id])
    for model in ('logistic','hist_gradient_boosting'):
        for fold,(train,held) in enumerate(splits):
            train_p=set(meta.iloc[train].paper_id);held_p=set(meta.iloc[held].paper_id)
            check(not train_p&held_p,'Outer paper overlap')
            _,inner=controller.paper_folds(meta.iloc[train].reset_index(drop=True),4)
            for j,(tr,va) in enumerate(inner):
                train_ix=train[tr];valid_ix=train[va]
                tp=set(meta.iloc[train_ix].paper_id);vp=set(meta.iloc[valid_ix].paper_id)
                a=keyed[model,fold,j]
                check(set(a['training_papers'])==tp and set(a['validation_papers'])==vp and set(a['outer_heldout_papers'])==held_p,
                    'Saved inner membership differs from frozen nested splitting')
                check(tp|vp==train_p,'Inner groups do not cover outer training papers')
                overlaps=dict(outer_train_heldout_overlap=len(train_p&held_p),inner_train_validation_overlap=len(tp&vp),
                    inner_train_outer_heldout_overlap=len(tp&held_p),inner_validation_outer_heldout_overlap=len(vp&held_p))
                check(not any(overlaps.values()),'Nested paper leakage')
                records.append(dict(model=model,outer_fold=fold,inner_fold=j,outer_train_queries=len(train),outer_heldout_queries=len(held),
                    inner_train_queries=len(tr),inner_validation_queries=len(va),**overlaps))
                check(len(set(y[train_ix]))==2,'Unexpected single-class learner fallback in actual inner FIT partition')
            check(len(set(y[train]))==2,'Unexpected single-class outer learner fallback')
    return folds,splits,records


def audit_thresholds(case,splits):
    table=csv_rows(case['oof']/'threshold_selection.csv');outcomes=csv_rows(case['oof']/'oof_predictions.csv')
    exports=[];coverage=[]
    for model in ('logistic','hist_gradient_boosting'):
        g=[r for r in outcomes if r['model']==model];by={r['question_id']:r for r in g}
        check(len(g)==len(by)==1746 and set(by)==set(case['meta'].question_id),'OOF duplicate/missing prediction')
        for fold,(train,held) in enumerate(splits):
            grid=[r for r in table if r['model']==model and int(r['outer_fold'])==fold]
            check(len(grid)==len(case['cfg']['thresholds']) and sorted(float(r['threshold']) for r in grid)==case['cfg']['thresholds'],'Inner threshold grid changed')
            check(all(r['source']=='inner-paper-OOF' and int(r['n_queries'])==len(train) for r in grid),'Inner threshold uses wrong sample/source metadata')
            selected,best,eligible=pick_threshold(grid)
            threshold=float(selected['threshold'])
            for i in held:
                m=case['meta'].iloc[i];r=by[m.question_id]
                check(r['paper_id']==m.paper_id and int(r['outer_fold'])==fold,'OOF prediction paper/fold mismatch')
                probability=float(r['predicted_accept_probability']);check(0<=probability<=1,'Invalid OOF probability')
                check((r['accepted']=='True')==(probability>=threshold),'Outer OOF decisions do not use inner-selected threshold')
            exports.append(dict(model=model,outer_fold=fold,selection_source='inner OOF from outer training only',
                selected_threshold=threshold,inner_best_f1=best,selected_inner_f1=float(selected['f1']),
                eligible_thresholds=json.dumps([float(r['threshold']) for r in eligible]),hurt=float(selected['hurt']),
                acceptance_rate=float(selected['acceptance_rate']),outer_train_queries=len(train),outer_heldout_queries=len(held)))
            coverage.append(dict(model=model,outer_fold=fold,expected_predictions=len(held),actual_predictions=sum(int(r['outer_fold'])==fold for r in g)))
        pooled=[r for r in table if r['model']==model and int(r['outer_fold'])==-1]
        selected,best,eligible=pick_threshold(pooled)
        deployment=json.loads((case['oof']/f'{model}_deployment.json').read_text())
        check(deployment['threshold']==float(selected['threshold']) and deployment['post_selection'],'Deployment threshold identity changed')
        if model=='logistic':check(deployment['threshold']==.70,'Previously selected logistic deployment threshold is not .70')
        check(deployment['feature_columns']==list(case['x'].columns) and deployment['model_identity_hash']==digest(case['p']),'Deployment schema/source identity changed')
        for grid_row in pooled:
            threshold=float(grid_row['threshold']);selected_scores=[];helps=[];hurts=[];accepts=[]
            for r in g:
                q=r['question_id'];take=float(r['predicted_accept_probability'])>=threshold
                base=case['bundle']['groups']['D0'][q];fresh=case['bundle']['groups']['F0'][q]
                score=fresh['f1'] if take else base['f1'];selected_scores.append(score)
                helps.append(score-base['f1']>.01);hurts.append(score-base['f1']<-.01);accepts.append(take)
            for key,values in [('f1',selected_scores),('help',helps),('hurt',hurts),('acceptance_rate',accepts)]:
                close(np.mean(values),grid_row[key],f'Pooled grid {key} mismatch')
        exports.append(dict(model=model,outer_fold=-1,selection_source='pooled outer OOF: deployment only, post-selection',
            selected_threshold=float(selected['threshold']),inner_best_f1=None,selected_inner_f1=None,
            eligible_thresholds=json.dumps([float(r['threshold']) for r in eligible]),hurt=float(selected['hurt']),
            acceptance_rate=float(selected['acceptance_rate']),pooled_selected_f1=float(selected['f1']),
            outer_train_queries=1746,outer_heldout_queries=0))
    return outcomes,exports,coverage


def independently_measure(case,outcomes):
    summaries=[];g=case['bundle']['groups'];by_model={}
    for model in ('logistic','hist_gradient_boosting'):
        rows=[r for r in outcomes if r['model']==model];scored=[]
        for r in rows:
            q=r['question_id'];base=g['D0'][q];fresh=g['F0'][q];accepted=r['accepted']=='True'
            expected=fresh if accepted else base
            check(r['prediction']==expected['prediction'],'Selected answer string differs')
            for key in ('f1','exact_match'):close(r[key],expected[key],f'Selected {key} differs')
            delta=expected['f1']-base['f1'];action_delta=fresh['f1']-base['f1']
            close(r['delta_f1'],delta,'Selected action delta differs');close(r['action_delta'],action_delta,'Proposed action delta differs')
            check((r['help']=='True')==(delta>.01) and (r['hurt']=='True')==(delta<-.01),'OOF HELP/HURT differs')
            check(r['action_state']==('HELP' if action_delta>.01 else 'HURT' if action_delta<-.01 else 'NEUTRAL'),'Proposed action class differs')
            for key in ('final_prompt_tokens','total_query_prompt_tokens','total_query_generated_tokens','num_lm_calls','revision_latency'):
                close(r[key],fresh[key],f'KEEP/ACCEPT did not retain full F0 cost: {key}')
            check(r['measured_end_to_end_latency']=='','Invented end-to-end time')
            scored.append(dict(question_id=q,paper_id=r['paper_id'],f1=expected['f1'],exact_match=expected['exact_match'],
                accepted=accepted,delta=delta,action_delta=action_delta,help=delta>.01,hurt=delta<-.01))
        by_model[model]={r['question_id']:r for r in scored}
        accepted=[r for r in scored if r['accepted']];helped=[r for r in scored if r['action_delta']>.01]
        hurt=[r for r in scored if r['action_delta']<-.01];neutral=[r for r in scored if abs(r['action_delta'])<=.01]
        summary=dict(model=model,n_queries=len(scored),f1=float(np.mean([r['f1'] for r in scored])),
            exact_match=float(np.mean([r['exact_match'] for r in scored])),mean_selected_action_delta=float(np.mean([r['delta'] for r in scored])),
            acceptance_rate=len(accepted)/1746,help=sum(r['help'] for r in scored)/1746,hurt=sum(r['hurt'] for r in scored)/1746,
            harm_rejection_rate=sum(not r['accepted'] for r in hurt)/len(hurt),
            help_retention_rate=sum(r['accepted'] for r in helped)/len(helped),missed_help_rate=sum(not r['accepted'] for r in helped)/len(helped),
            false_accept_rate=sum(r['action_delta']<=.01 for r in accepted)/len(accepted),
            harmful_accept_precision=sum(r['action_delta']<-.01 for r in accepted)/len(accepted),
            neutral_accept_rate=sum(r['accepted'] for r in neutral)/len(neutral),
            accept_on_help=sum(r['accepted'] for r in helped),keep_on_help=sum(not r['accepted'] for r in helped),
            accept_on_hurt=sum(r['accepted'] for r in hurt),keep_on_hurt=sum(not r['accepted'] for r in hurt),
            accepted_neutral_count=sum(r['accepted'] for r in neutral))
        saved=next(r for r in csv_rows(case['oof']/'summary.csv') if r['model']==model)
        for key,value in summary.items():
            if key in saved and key!='model':close(value,saved[key],f'OOF summary {model}/{key} mismatch')
        summaries.append(summary)
    return summaries,by_model


def run():
    case=load_case();directory=case['root']/'audit_pre_final';directory.mkdir(parents=True,exist_ok=True)
    lineage=audit_lineage(case);feature_rows=feature_audit(case);models=audit_models(case)
    folds,splits,overlaps=audit_folds(case);outcomes,thresholds,coverage=audit_thresholds(case,splits)
    measured,by_model=independently_measure(case,outcomes)
    boot=[];saved_boot=csv_rows(case['oof']/'bootstrap.csv')
    for baseline in ('D0','F0','F2-A'):
        result=bootstrap(by_model['logistic'],case['bundle']['groups'][baseline])
        if baseline!='F2-A':
            saved=next(r for r in saved_boot if r['model']=='logistic' and r['baseline']==baseline)
            for key in ('point_estimate','ci95_low','ci95_high'):close(result[key],saved[key],'Independent bootstrap mismatch')
        boot.append(dict(comparison='Logistic - '+baseline,post_selection=True,descriptive=True,
            preregistered_comparison=baseline!='F2-A',**result))
    tables={'action_lineage.csv':lineage,'feature_leakage_and_missingness.csv':feature_rows,
        'fold_overlap_counts.csv':overlaps,'prediction_coverage.csv':coverage,
        'selected_thresholds.csv':thresholds,'independent_oof_metrics.csv':measured,'independent_bootstrap.csv':boot}
    for name,records in tables.items():pd.DataFrame(records).to_csv(directory/name,index=False)
    atomic_json(directory/'model_parameter_checks.json',models)
    from .trace import synthetic_trace
    atomic_json(directory/'preprocessing_call_trace.json',synthetic_trace())
    classification=dict(HELP=sum(r['state']=='HELP' for r in lineage),NEUTRAL=sum(r['state']=='NEUTRAL' for r in lineage),HURT=sum(r['state']=='HURT' for r in lineage))
    atomic_json(directory/'runtime_and_provenance_limits.json',dict(
        candidate_latency='Saved F0 generation latency, inherited per query; not a new generation measurement',
        prediction_latency='Recorded batch prediction time amortized per outer-heldout query',
        feature_computation_latency='Not measured separately; unknown, not zero',
        preprocessing_latency='Not measured separately',measured_end_to_end_latency=None,
        inner_probabilities='Not archived. Inner grid cannot be independently rescored; verify source/data access, saved partitions, grid metadata and decisions instead.',
        fold_estimators='Not archived; final full-FIT pipelines are archived. Historical per-fold fitted statistics cannot be recovered without forbidden unnecessary OOF retraining.',
        existing_full_fit='OOF runner already fits final estimators on all FIT; not used for OOF predictions. Export/refit code must retain this distinction.'))
    protocol.verify_hashes(case['protected'])
    # The synthetic call-path/preprocessing trace is produced by separate tests,
    # not by fitting a real learner during this audit.
    summary=dict(status='VALID WITH NON-MATERIAL FIXES',oof_rerun_required=False,
        safe_for_final_fit=True,queries=1746,papers=656,features=32,sklearn=case['p']['sklearn'],
        target_counts=classification,source_controller_protocol_hash=digest(case['p']),
        frozen_feature_columns=list(case['x'].columns),selected_model='logistic',deployment_threshold=.70,
        no_new_features_or_models=True,no_real_training_performed=True,test_touched=False,
        material_issues=[],non_material_issues=[
            'Selected outer-fold thresholds were implicit in saved grids; exported explicit selection metadata without changing decisions.',
            'Full-FIT models already exist inside controller_fit_oof; separate final packaging/preparation is added to prevent confusion with honest OOF.',
            'Feature-computation latency and raw inner probabilities/fold estimators were not archived; explicitly report unknowns and audit evidence limits.'],
        input_hashes=case['protected'],original_artifacts_unchanged=True,
        audit_code_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in Path(__file__).parent.glob('*.py')})
    from .report import write
    report_path=write(case,summary)
    summary['report_path']=str(report_path)
    summary['artifact_hashes']={str(p.resolve()):file_hash(p) for p in directory.iterdir() if p.is_file() and p.name!='audit_status.json'}
    summary['artifact_hashes'][str(report_path)]=file_hash(report_path)
    atomic_json(directory/'audit_status.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('input_hashes','audit_code_hashes','frozen_feature_columns','artifact_hashes')},indent=2))
    return case,summary


if __name__=='__main__':run()
