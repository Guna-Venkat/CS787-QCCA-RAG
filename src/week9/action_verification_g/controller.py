"""Disabled-by-default G2 ladder; nested paper OOF when explicitly enabled later."""
import json
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from ..provenance import ROOT,branch_guard,digest,file_hash
from ..caching import atomic_json
from .protocol import resolve,pin,verify_hashes
from .preflight import target
from .features import prepare,AVAILABILITY,GROUPS


def paper_folds(meta,n=5):
    from sklearn.model_selection import GroupKFold
    if meta.duplicated(['question_id']).any() or meta.paper_id.nunique()<n:
        raise ValueError('Paper OOF requires unique queries and enough distinct papers')
    assignments=np.full(len(meta),-1,int);splits=[]
    for fold,(train,valid) in enumerate(GroupKFold(n).split(meta,groups=meta.paper_id)):
        if set(meta.iloc[train].paper_id)&set(meta.iloc[valid].paper_id):raise ValueError('Paper leakage')
        assignments[valid]=fold;splits.append((train,valid))
    if (assignments<0).any():raise ValueError('Incomplete held-out assignment')
    return assignments,splits


def applied_outcomes(groups,meta,accept):
    if len(meta)!=len(accept):raise ValueError('Prediction/action alignment mismatch')
    rows=[]
    for (_,m),take in zip(meta.iterrows(),accept):
        q=m.question_id;base=groups['D0'][q];candidate=groups['F0'][q];r=candidate if take else base
        if m.paper_id!=base['paper_id'] or m.paper_id!=candidate['paper_id']:raise ValueError('Action paper mismatch')
        delta=candidate['f1']-base['f1'];selected_delta=r['f1']-base['f1']
        rows.append(dict(question_id=q,paper_id=m.paper_id,accepted=bool(take),
            prediction=r['prediction'],f1=r['f1'],exact_match=r['exact_match'],delta_f1=selected_delta,
            help=selected_delta>.01,hurt=selected_delta<-.01,action_delta=delta,
            action_state='HELP' if delta>.01 else 'HURT' if delta<-.01 else 'NEUTRAL',
            # Candidate was generated before the decision, even for KEEP.
            final_prompt_tokens=candidate['final_prompt_tokens'],
            total_query_prompt_tokens=candidate['total_query_prompt_tokens'],
            total_query_generated_tokens=candidate['total_query_generated_tokens'],num_lm_calls=candidate['num_lm_calls'],
            revision_latency=candidate['revision_latency'],measured_end_to_end_latency=None))
    return rows


def metrics(rows):
    def rate(fn,g):return float(np.mean([fn(r) for r in g])) if g else None
    h=[r for r in rows if r['action_state']=='HELP'];bad=[r for r in rows if r['action_state']=='HURT']
    neutral=[r for r in rows if r['action_state']=='NEUTRAL'];accepted=[r for r in rows if r['accepted']]
    return dict(n_queries=len(rows),f1=rate(lambda r:r['f1'],rows),exact_match=rate(lambda r:r['exact_match'],rows),
        mean_selected_action_delta=rate(lambda r:r['delta_f1'],rows),
        help=rate(lambda r:r['help'],rows),hurt=rate(lambda r:r['hurt'],rows),acceptance_rate=rate(lambda r:r['accepted'],rows),
        false_accept_rate=rate(lambda r:r['action_state']!='HELP',accepted),
        harmful_accept_precision=rate(lambda r:r['action_state']=='HURT',accepted),
        missed_help_rate=rate(lambda r:not r['accepted'],h),harm_rejection_rate=rate(lambda r:not r['accepted'],bad),
        help_retention_rate=rate(lambda r:r['accepted'],h),neutral_accept_rate=rate(lambda r:r['accepted'],neutral),
        accept_on_help=sum(r['accepted'] for r in h),keep_on_help=sum(not r['accepted'] for r in h),
        accept_on_hurt=sum(r['accepted'] for r in bad),keep_on_hurt=sum(not r['accepted'] for r in bad),
        mean_candidate_revision_latency=rate(lambda r:r['revision_latency'],rows),
        mean_verifier_predict_seconds=rate(lambda r:r['verifier_predict_seconds'],[r for r in rows if r.get('verifier_predict_seconds') is not None]),
        **{k:rate(lambda r:r[k],rows) for k in ('total_query_prompt_tokens','total_query_generated_tokens','num_lm_calls')})


def choose_threshold(probabilities,groups,meta,cfg):
    if not np.isfinite(probabilities).all():raise ValueError('Nonfinite OOF probability')
    table=[]
    for threshold in cfg['thresholds']:
        row=metrics(applied_outcomes(groups,meta,np.asarray(probabilities)>=threshold))
        table.append(dict(threshold=threshold,**row))
    best=max(r['f1'] for r in table)
    eligible=[r for r in table if r['f1']>=best-cfg['quality_tolerance']-1e-12]
    chosen=min(eligible,key=lambda r:(r['hurt'],r['acceptance_rate'],r['total_query_prompt_tokens'],abs(r['threshold']-.5),-r['threshold']))
    return chosen,table


def estimator(name,seed):
    # This factory does not fit. Imports remain lazy and CPU-only.
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    if name=='logistic':
        model=LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=seed)
        return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),model)
    if name=='hist_gradient_boosting':
        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,max_depth=3,
            min_samples_leaf=20,l2_regularization=1.,random_state=seed)
        return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),model)
    raise ValueError('Only two preregistered lightweight classifiers')


def fit_model(name,seed,x,y):
    if len(set(y))<2:
        from sklearn.dummy import DummyClassifier
        model=DummyClassifier(strategy='prior')
    else:model=estimator(name,seed)
    model.fit(x,y)
    return model


def predict_accept(model,x):
    p=model.predict_proba(x);classes=list(model.classes_)
    return p[:,classes.index(1)] if 1 in classes else np.zeros(len(x))


def verify_preflight(cfg,ident):
    directory=resolve(cfg['output_root'])/'preflight'
    manifest=json.loads((directory/'preflight_manifest.json').read_text())
    if not manifest['complete'] or manifest['protocol_hash']!=digest(ident):raise ValueError('Evaluate matching saved preflight first')
    verify_hashes(manifest['artifact_hashes'])
    decision=json.loads((directory/'go_no_go.json').read_text())
    if decision['recommended_route']!='G2' or decision['G2']['decision']!='GO':
        raise ValueError('This scaffold is G2-only and requires a matching G2 GO preflight')
    return manifest


def scaffold(cfg,bundle,ident):
    """Cheap features/fold assignments only. Never constructs/fits an estimator."""
    verify_preflight(cfg,ident)
    x,meta=prepare(bundle,cfg);folds,_=paper_folds(meta,cfg['outer_folds'])
    directory=resolve(cfg['output_root'])/'scaffold';directory.mkdir(parents=True,exist_ok=True)
    pd.concat([meta,x],axis=1).to_csv(directory/'inference_features.csv',index=False)
    meta.assign(outer_fold=folds).to_csv(directory/'fold_assignments.csv',index=False)
    atomic_json(directory/'feature_schema.json',dict(feature_columns=list(x.columns),groups=GROUPS,availability=AVAILABILITY,
        excluded_metadata_columns=['question_id','paper_id'],no_outcomes_in_features=True,
        missing_values='Absent rank/length features remain missing; imputation must fit only within training folds.'))
    atomic_json(directory/'manifest.json',dict(queries=len(meta),papers=meta.paper_id.nunique(),
        outer_folds=cfg['outer_folds'],controller_training_started=False,lm_calls=0,
        route='G2',protocol_hash=digest(ident),models=cfg['models'],
        oof='Outer paper groups=5, inner paper groups=4 for threshold selection; no query-level split.',
        notes='Features/folds only; no fitted estimator, OOF prediction or selected threshold exists.'))
    return x,meta


def train_manually(cfg,bundle,ident,enabled=False,resume=False):
    if not enabled:raise PermissionError('Controller training requires explicit opt-in after preflight review')
    branch_guard();verify_preflight(cfg,ident)
    x,meta=prepare(bundle,cfg)
    labels=np.array([target(bundle['groups']['D0'][q]['f1'],bundle['groups']['F0'][q]['f1'],cfg['epsilon'])['accept_f0'] for q in meta.question_id])
    directory=resolve(cfg['output_root'])/'controller_fit_oof';directory.mkdir(parents=True,exist_ok=True)
    own={str(p.relative_to(ROOT)):file_hash(p) for p in (Path(__file__),Path(__file__).with_name('features.py'))}
    model_ident=dict(preflight_protocol_hash=digest(ident),sources=own,config=cfg,features=list(x.columns),
        sklearn=version('scikit-learn'),target='THIS saved F0 delta > .01',
        oof='Outer5 paper OOF; inner4 paper OOF within outer training selects threshold; scalers/imputers fit only on training rows.')
    pin(directory/'protocol.json',model_ident)
    if (directory/'manifest.json').exists():
        if not resume:raise FileExistsError('Completed training exists; use --resume or a new version')
        manifest=json.loads((directory/'manifest.json').read_text());verify_hashes(manifest['artifact_hashes'])
        return manifest
    fold_ids,splits=paper_folds(meta,cfg['outer_folds']);all_summary=[];all_outputs={};thresholds=[];fold_audits=[]
    for name in cfg['models']:
        probabilities=np.full(len(meta),np.nan);accepted=np.zeros(len(meta),bool);prediction_seconds=np.zeros(len(meta))
        for fold,(train,heldout) in enumerate(splits):
            train_meta=meta.iloc[train].reset_index(drop=True)
            inner_prob=np.full(len(train),np.nan)
            _,inner_splits=paper_folds(train_meta,cfg['inner_folds'])
            for j,(inner_train,inner_valid) in enumerate(inner_splits):
                tr=train[inner_train];va=train[inner_valid]
                model=fit_model(name,cfg['seed']+fold*10+j,x.iloc[tr],labels[tr])
                inner_prob[inner_valid]=predict_accept(model,x.iloc[va])
                fold_audits.append(dict(model=name,outer_fold=fold,inner_fold=j,
                    training_papers=sorted(meta.iloc[tr].paper_id.unique()),validation_papers=sorted(meta.iloc[va].paper_id.unique()),
                    outer_heldout_papers=sorted(meta.iloc[heldout].paper_id.unique())))
            chosen,grid=choose_threshold(inner_prob,bundle['groups'],train_meta,cfg)
            thresholds.extend(dict(model=name,outer_fold=fold,source='inner-paper-OOF',**r) for r in grid)
            model=fit_model(name,cfg['seed']+fold,x.iloc[train],labels[train])
            started=time.perf_counter();prob=predict_accept(model,x.iloc[heldout]);latency=time.perf_counter()-started
            probabilities[heldout]=prob;accepted[heldout]=prob>=chosen['threshold'];prediction_seconds[heldout]=latency/len(heldout)
        if not np.isfinite(probabilities).all():raise ValueError('Incomplete outer OOF predictions')
        outcomes=applied_outcomes(bundle['groups'],meta,accepted)
        for i,r in enumerate(outcomes):r.update(model=name,outer_fold=int(fold_ids[i]),predicted_accept_probability=float(probabilities[i]),
            verifier_predict_seconds=float(prediction_seconds[i]))
        all_outputs[name]=outcomes
        all_summary.append(dict(model=name,evaluation='nested-paper-OOF',**metrics(outcomes)))
        # Deployment threshold uses all outer OOF probabilities; not the source
        # of the honest nested-threshold evaluation above.
        deploy,grid=choose_threshold(probabilities,bundle['groups'],meta,cfg)
        thresholds.extend(dict(model=name,outer_fold=-1,source='all-outer-OOF deployment selection; post-selection',**r) for r in grid)
        final=fit_model(name,cfg['seed'],x,labels)
        import joblib
        joblib.dump(final,directory/f'{name}.joblib')
        pin(directory/f'{name}_deployment.json',dict(model=name,threshold=deploy['threshold'],feature_columns=list(x.columns),
            threshold_selection='All FIT outer OOF probabilities; final estimator fitted on FIT after honest nested evaluation.',
            quality_tolerance=.003,tie_rule='Lower HURT, acceptance, candidate cost, then closest to .5 and higher threshold.',
            post_selection=True,model_identity_hash=digest(model_ident)))
    pd.DataFrame(all_summary).to_csv(directory/'summary.csv',index=False)
    pd.DataFrame([r for v in all_outputs.values() for r in v]).to_csv(directory/'oof_predictions.csv',index=False)
    pd.DataFrame(thresholds).to_csv(directory/'threshold_selection.csv',index=False)
    meta.assign(outer_fold=fold_ids).to_csv(directory/'fold_assignments.csv',index=False)
    atomic_json(directory/'fold_audit.json',fold_audits)
    best=max(r['f1'] for r in all_summary);eligible=[r for r in all_summary if r['f1']>=best-.003-1e-12]
    selected=min(eligible,key=lambda r:(r['hurt'],r['acceptance_rate'],r['model']!='logistic'))
    atomic_json(directory/'frozen_verifier.json',dict(selected_model=selected['model'],metrics=selected,
        selected_on='FIT nested paper OOF; selecting best model is post-selection, not held-out confirmation',
        protocol_hash=digest(model_ident),action='KEEP_D0 vs ACCEPT_THIS_F0',candidate_cost_already_paid=True))
    from ..evaluation import paired_paper_bootstrap
    stats=[]
    for name,outcomes in all_outputs.items():
        for base in ('D0','F0'):
            result=paired_paper_bootstrap(outcomes,list(bundle['groups'][base].values()),n=10000,seed=42)
            stats.append(dict(model=name,baseline=base,point_estimate=result['point_estimate'],
                ci95_low=result['ci95'][0],ci95_high=result['ci95'][1],post_selection=name==selected['model']))
    pd.DataFrame(stats).to_csv(directory/'bootstrap.csv',index=False)
    verify_hashes(bundle['input_hashes'])
    manifest=dict(complete=True,split='fit',controller_training_started=True,lm_calls=0,
        selected_model=selected['model'],protocol_hash=digest(model_ident),
        measured_end_to_end_latency=None,cost_note='All queries pay saved F0 generation plus feature/verifier overhead, including KEEP outcomes.',
        artifact_hashes={str(p.resolve()):file_hash(p) for p in directory.iterdir() if p.is_file() and p.name!='manifest.json'})
    atomic_json(directory/'manifest.json',manifest)
    return manifest
