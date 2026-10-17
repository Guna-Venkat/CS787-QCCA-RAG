"""FIT-OOF controller construction and honest single-promotion cached replay.

Decisions consume predicted probabilities/features only. Outcome lookup is a
separate retrospective evaluation step; no oracle chooses PROMOTE versus STOP.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.week8.closeout import KEYS,cached_split,attach_saved_predictions,chosen_rows,heuristic_score
from src.week8.data import PROJECT_ROOT
from src.week8.gate_diagnostics import U_COLUMNS,gate_source_fingerprint
from src.week8.generator_features import source_fingerprint
from src.week8.generator_features import sha256_file
from src.week8.controller_costs import apply_exact_token_costs
from src.week6.evaluation import compute_qa_answer_metrics

THRESHOLDS=[0.,.1,.2,.3,.4,.5,.6,.8,1.01]
OUT=PROJECT_ROOT/'results/week8/w8e_controller'


def controller_sources():
    paths=[PROJECT_ROOT / name for name in [
        'src/week8/selective_controller.py','src/week8/closeout.py','src/week8/controller_costs.py',
        'src/week8/gate_diagnostics.py','src/week8/generator_features.py',
        'src/week8/utility_training.py','src/week8/utility_models.py',
        'src/week8/semantic.py','src/week8/marginal_information.py',
        'src/week7/w7e/prompt_builder.py','src/week6/routing.py',
        'results/week8/w8d_closeout/closeout_summary.json',
        'results/week8/w8d_closeout/fit_selector_scores.csv',
        'results/week8/w8d_closeout/internal_val_selector_scores.csv',
        'results/week8/w8c_utility/U4-A_fit_oof_predictions.csv',
        'results/week8/w8c_utility/selected_internal_val_predictions.csv',
        'results/week8/w8c_utility/selected_neural.pt',
        'results/week8/w8c_utility/w8c_frozen_selector_config.json',
        'results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl']]
    for split in ['fit','internal_val']:
        paths.extend(PROJECT_ROOT / name for name in [
            f'results/week8/w8e_gate_inputs/gate_features_{split}.csv',
            f'results/week8/w8e_gate_inputs/provenance_{split}.json',
            f'results/week8/surfaces/promotion_surface_{split}.jsonl'])
    return {str(p.relative_to(PROJECT_ROOT)):sha256_file(p) for p in paths}


def gate_inputs(df,split):
    folder=PROJECT_ROOT/'results/week8/w8e_gate_inputs'
    path=folder/f'gate_features_{split}.csv'
    if not path.is_file():raise FileNotFoundError(f'Missing measured FIT/VAL token uncertainty: {path}. Run backfill_w8e_gate_inputs.py.')
    fingerprint,_,_,_=gate_source_fingerprint(split)
    provenance=json.loads((folder/f'provenance_{split}.json').read_text())
    if provenance['source_fingerprint']!=fingerprint:raise ValueError('Gate feature provenance changed')
    if provenance.get('feature_csv_sha256')!=sha256_file(path):
        raise ValueError('Gate CSV integrity is unverified; run the measured-input backfill validator')
    features=pd.read_csv(path,dtype={'question_id':str,'paper_id':str})
    q=df.groupby('question_id',sort=True).first().reset_index()
    result=q[['question_id','paper_id','best_delta']].merge(features,on=['question_id','paper_id'],how='left',validate='one_to_one',sort=False)
    if len(result)!=len(q) or not np.isfinite(result[U_COLUMNS].to_numpy(float)).all():raise ValueError('Incomplete measured gate features')
    return result


def train_gain_weighted_gate(q):
    X=q[U_COLUMNS].to_numpy(float);y=(q.best_delta.to_numpy()>.01).astype(int)
    weights=1+5*np.maximum(q.best_delta.to_numpy(),0)
    def fit(tr):
        if len(np.unique(y[tr]))!=2:raise ValueError('Both needs-promotion classes required in FIT training fold')
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=42))
        model.fit(X[tr],y[tr],logisticregression__sample_weight=weights[tr]);return model
    oof=np.full(len(q),np.nan);fold_records=[]
    for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,q.paper_id)):
        if set(q.iloc[tr].paper_id)&set(q.iloc[va].paper_id):raise ValueError('Gate paper leakage')
        model=fit(tr);oof[va]=model.predict_proba(X[va])[:,1]
        fold_records.extend({'question_id':q.iloc[i].question_id,'paper_id':q.iloc[i].paper_id,'fold':fold} for i in va)
    if not np.isfinite(oof).all():raise ValueError('Incomplete gate OOF coverage')
    return oof,fit(np.arange(len(q))),fold_records


def make_decisions(candidates,gate_probabilities,config):
    """Pure online rule: no gold, F1, oracle label or true utility is read here."""
    required=KEYS+['bm25_rank','selection_score','prob_help','prob_hurt']
    missing=set(required)-set(candidates)
    if missing:raise ValueError(f'Missing actual candidate predictions: {sorted(missing)}')
    if candidates.duplicated(KEYS).any():raise ValueError('Duplicate candidate identities')
    if not np.isfinite(candidates[['selection_score','prob_help','prob_hurt']].to_numpy(float)).all():raise ValueError('Nonfinite candidate predictions')
    if not candidates[['prob_help','prob_hurt']].apply(lambda s:s.between(0,1).all()).all():raise ValueError('Invalid predicted probabilities')
    gate=pd.Series(gate_probabilities,dtype=float)
    if not set(candidates.question_id)<=set(gate.index):raise ValueError('Missing measured gate predictions')
    if not np.isfinite(gate.to_numpy()).all() or not gate.between(0,1).all():raise ValueError('Invalid gate probabilities')
    selected=candidates[required].sort_values('semantic_rank')
    selected=selected.loc[selected.groupby('question_id').selection_score.idxmax()].sort_values('question_id').copy()
    p=selected.question_id.map(gate).to_numpy()
    if config['family']=='E0':promote=np.zeros(len(selected),bool)
    elif config['family']=='E1':promote=np.ones(len(selected),bool)
    else:
        promote=(p>=config['gate_threshold'])
        promote&=selected.prob_help.to_numpy()>config.get('tau_help',-1.)
        promote&=selected.prob_hurt.to_numpy()<config.get('tau_hurt',1.01)
    selected['selected_action']=np.where(promote,'PROMOTE','STOP')
    selected['gate_probability']=p
    for column in ['candidate_passage_index','semantic_rank','bm25_rank']:
        selected['proposed_'+column]=selected[column].to_numpy()
        selected[column]=selected[column].where(promote,pd.NA).astype('Int64')
    selected['decision_reason']=np.where(promote,'predicted_gate_and_safety_passed','gate_or_safety_rejected')
    if config['family']=='E0':selected['decision_reason']='baseline_return_draft'
    if config['family']=='E1':selected['decision_reason']='always_promote'
    selected['policy_name']=config['policy_name']
    selected['uses_attribution']=bool(config.get('uses_attribution',False))
    return selected.drop(columns=['selection_score']).reset_index(drop=True)


def canonical_baselines(df):
    qids=set(df.question_id);rows=[]
    path=PROJECT_ROOT/'results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl'
    with path.open() as f:
        for line in f:
            r=json.loads(line)
            if str(r['question_id']) in qids and int(r['k']) in (4,6):
                rows.append({'question_id':str(r['question_id']),'k':int(r['k']),
                    'f1':compute_qa_answer_metrics(r['prediction'],r['gold_answers'],r.get('gold_answers_reformatted',r['gold_answers']))['token_f1'],
                    'tokens':float(r['context_tokens']),'generation_latency':float(r['latency'])})
    base=pd.DataFrame(rows)
    if len(base)!=2*len(qids) or base.duplicated(['question_id','k']).any():raise ValueError('Incomplete canonical k4/k6 baselines')
    if 'exact_k6_prompt_tokens' in df:
        costs=df.groupby('question_id').first()
        for k,column in [(4,'exact_base_prompt_tokens'),(6,'exact_k6_prompt_tokens')]:
            selected=base.k==k
            base.loc[selected,'tokens']=base.loc[selected,'question_id'].map(costs[column])
    return base


def evaluate_decisions(decisions,surface,baselines):
    """Attach actual saved outcomes AFTER decisions, including harmed promotions."""
    columns=KEYS+['delta_f1','promoted_f1','promoted_context_tokens','promotion_latency']
    exact_base='exact_base_prompt_tokens' in surface
    exact_promotion='exact_promoted_prompt_tokens' in surface
    if exact_base!=exact_promotion:raise ValueError('Incomplete exact prompt-token accounting')
    if exact_promotion:columns.append('exact_promoted_prompt_tokens')
    chosen=decisions.merge(surface[columns],on=KEYS,how='left',validate='one_to_one')
    query_columns=['base_f1','base_context_tokens','best_delta']
    if exact_base:query_columns.append('exact_base_prompt_tokens')
    if 'feature_extraction_seconds_query' in surface:
        query_columns.append('feature_extraction_seconds_query')
    q=surface.groupby('question_id').first()[query_columns]
    chosen=chosen.merge(q,left_on='question_id',right_index=True,how='left',validate='one_to_one')
    base=baselines[baselines.k==4].set_index('question_id')
    k6=baselines[baselines.k==6].set_index('question_id')
    promote=chosen.selected_action.eq('PROMOTE')
    if chosen.loc[promote,['delta_f1','promoted_f1','promotion_latency']].isna().any().any():raise ValueError('Selected passage has no measured outcome')
    chosen['realized_delta_f1']=chosen.delta_f1.where(promote,0.)
    chosen['final_f1']=chosen.promoted_f1.where(promote,chosen.base_f1)
    base_cost=chosen.exact_base_prompt_tokens if exact_base else chosen.base_context_tokens
    promotion_cost=chosen.exact_promoted_prompt_tokens if exact_promotion else chosen.promoted_context_tokens
    chosen['final_prompt_tokens']=promotion_cost.where(promote,base_cost)
    chosen['cumulative_prompt_tokens']=base_cost+promotion_cost.where(promote,0.)
    chosen['estimated_generation_latency_sec']=chosen.question_id.map(base.generation_latency)+chosen.promotion_latency.where(promote,0.)
    needs_attribution=promote&chosen.uses_attribution
    if needs_attribution.any() and 'feature_extraction_seconds_query' not in chosen:
        raise ValueError('Missing measured attribution cost for attribution-based controller')
    chosen['estimated_attribution_latency_sec']=0.
    if needs_attribution.any():
        costs=chosen.loc[needs_attribution,'feature_extraction_seconds_query']
        if not np.isfinite(costs).all():raise ValueError('Nonfinite measured attribution costs')
        chosen.loc[needs_attribution,'estimated_attribution_latency_sec']=costs
    chosen['estimated_total_recorded_latency_sec']=chosen.estimated_generation_latency_sec+chosen.estimated_attribution_latency_sec
    n=len(chosen);num=int(promote.sum());hurt=chosen.realized_delta_f1<-.01;helpful=chosen.realized_delta_f1>.01
    opportunity=chosen.best_delta>.01
    metrics={'num_queries':n,'promotion_rate':float(promote.mean()),'num_promotions':num,
        'downstream_f1':float(chosen.final_f1.mean()),'gain_over_k4':float((chosen.final_f1-chosen.base_f1).mean()),
        'gain_over_k6':float((chosen.final_f1-chosen.question_id.map(k6.f1)).mean()),
        'mean_final_prompt_tokens':float(chosen.final_prompt_tokens.mean()),
        'mean_cumulative_prompt_tokens':float(chosen.cumulative_prompt_tokens.mean()),
        'estimated_generation_latency_sec':float(chosen.estimated_generation_latency_sec.mean()),
        'estimated_attribution_latency_sec':float(chosen.estimated_attribution_latency_sec.mean()),
        'estimated_total_recorded_latency_sec':float(chosen.estimated_total_recorded_latency_sec.mean()),
        'end_to_end_latency_sec':None,
        'help_rate_all_queries':float(helpful.mean()),'hurt_rate_all_queries':float(hurt.mean()),
        'help_precision_promotions':float(helpful.sum()/num) if num else 0.,
        'hurt_rate_promotions':float(hurt.sum()/num) if num else 0.,
        'escalation_recall':float((promote&opportunity).sum()/opportunity.sum()) if opportunity.any() else 0.,
        'unnecessary_promotion_rate':float((promote&~opportunity).mean()),
        'latency_note':'Cached k4 generation + selected W8-A promotion + measured attribution extraction when required; excludes other unmeasured routing, gate, and selector overhead. Not measured end-to-end deployment latency.'}
    return metrics,chosen


def choose_config(records,tolerance=.003):
    if not records:raise ValueError('No real controller configurations evaluated')
    best=max(r['metrics']['downstream_f1'] for r in records)
    eligible=[r for r in records if r['metrics']['downstream_f1']>=best-tolerance]
    return min(eligible,key=lambda r:(r['metrics']['mean_cumulative_prompt_tokens'],
        r['metrics'].get('estimated_total_recorded_latency_sec',r['metrics']['estimated_generation_latency_sec']),r['config']['complexity'],r['config']['policy_name'],
        r['config'].get('gate_threshold',0),r['config'].get('tau_help',-1),r['config'].get('tau_hurt',1.01)))


def choose_family_operating_points(records_by_family):
    """Apply the final quality tolerance once across the entire real grid.

    Taking family representatives first and applying the tolerance again can
    select a policy more than 0.003 below the true global best.
    """
    representatives={name:choose_config(records) for name,records in records_by_family.items()}
    selected=choose_config([record for records in records_by_family.values() for record in records])
    representatives[selected['config']['policy_name']]=selected
    return selected,representatives


def candidate_table(df,scores,prob_help,prob_hurt):
    result=df[KEYS+['bm25_rank']].copy()
    result['selection_score']=np.asarray(scores);result['prob_help']=np.asarray(prob_help);result['prob_hurt']=np.asarray(prob_hurt)
    return result


def _policy_grid(name,selector,complexity,veto=False):
    if name in ('E0','E1'):
        return [{'policy_name':name,'family':name,'selector':selector,'complexity':complexity}]
    return [{'policy_name':name,'family':'E5' if veto else name,'selector':selector,'complexity':complexity,
        'gate_threshold':float(t),'tau_help':float(h),'tau_hurt':float(d)}
        for t in THRESHOLDS for h in ([0.,.1,.2] if veto else [-1.])
        for d in ([1.01,.2,.1] if veto else [1.01])]


def run_controller(output_dir=OUT):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    sources=controller_sources()
    if (output_dir/'controller_summary.json').exists():
        saved=json.loads((output_dir/'controller_summary.json').read_text())
        if saved.get('source_hashes')!=sources:
            raise ValueError('Completed controller inputs/code changed. Inspect provenance; do not overwrite the frozen run.')
        return saved
    close=PROJECT_ROOT/'results/week8/w8d_closeout'
    summary=json.loads((close/'closeout_summary.json').read_text())
    # Tune parameters on FIT only. Earlier closeout confirmation controls E4
    # eligibility; controller validation scores are computed after FIT freeze.
    fit=apply_exact_token_costs(cached_split('fit'),'fit');fit_gate=gate_inputs(fit,'fit');base=canonical_baselines(fit)
    probabilities,gate,fold_records=train_gain_weighted_gate(fit_gate)
    pd.DataFrame(fold_records).to_csv(output_dir/'gate_paper_folds.csv',index=False)
    fit_prob=dict(zip(fit_gate.question_id,probabilities))
    joblib.dump(gate,output_dir/'b0_gw_gate.joblib')
    pd.DataFrame({'question_id':fit_gate.question_id,'paper_id':fit_gate.paper_id,
        'prob_needs_promotion':probabilities}).to_csv(output_dir/'fit_gate_oof_predictions.csv',index=False)
    saved=attach_saved_predictions(fit,'fit')
    score=pd.read_csv(close/'fit_selector_scores.csv',dtype={'question_id':str,'paper_id':str})
    score=fit[KEYS].merge(score,on=KEYS,validate='one_to_one',how='left')
    if score.S3.isna().any():raise ValueError('Missing FIT closeout predictions')
    candidate_scores={'S3':score.S3.to_numpy(),'U4-A':saved['utility_score']}
    eligible_attr=summary['decision']['online_attribution_survives']
    if eligible_attr:
        name=summary['decision']['candidate_frozen_on_fit'];candidate_scores[name]=score[name].to_numpy()
    tables={n:candidate_table(fit,s,saved['prob_help'],saved['prob_hurt']) for n,s in candidate_scores.items()}
    families=[('E0','S3',0,False),('E1','S3',1,False),('E2','S3',2,False),('E3','U4-A',3,False),
        ('E5-DraftSemantic','S3',4,True),('E5-U4-A','U4-A',5,True)]
    if eligible_attr:families.append(('E4',summary['decision']['candidate_frozen_on_fit'],6,False))
    all_records=[];records_by_family={}
    for name,selector,complexity,veto in families:
        records=[]
        for config in _policy_grid(name,selector,complexity,veto):
            config['uses_attribution']=bool(name=='E4')
            decisions=make_decisions(tables[selector],fit_prob,config)
            metrics,_=evaluate_decisions(decisions,fit,base)
            records.append({'config':config,'metrics':metrics})
        records_by_family[name]=records;all_records.extend(records)
    selected,family_best=choose_family_operating_points(records_by_family)
    grid_best=max(record['metrics']['downstream_f1'] for record in all_records)
    shortfall=grid_best-selected['metrics']['downstream_f1']
    if shortfall>.003+1e-12:
        raise AssertionError('Selected controller is outside the global 0.003 quality tolerance')
    frozen={'gate':'B0-GW logistic: measured token U1-U12; weight=1+5*max(0,best_delta)',
        'gate_training':'5-fold paper-grouped FIT OOF; full FIT refit',
        'selection_rule':'highest FIT OOF downstream F1; within .003 lower cumulative prompt tokens, then recorded generation/attribution latency estimate, then simplicity',
        'family_configurations':{n:r['config'] for n,r in family_best.items()},
        'selected_policy':selected['config']['policy_name'],'selected_config':selected['config'],
        'fit_grid_best_downstream_f1':grid_best,'fit_selected_shortfall_from_grid_best':shortfall,
        'family_rows_note':'Per-family cost-efficient operating points; the overall selected point is chosen once from the full grid and replaces its family row.',
        'fit_metrics':{n:r['metrics'] for n,r in family_best.items()},
        'E4_included':bool(eligible_attr),'E4_reason':summary['decision']['verdict'],
        'max_promotions':1,'DAFiR2':'not evaluated; requires sequential counterfactual outcomes',
        'frozen_before_internal_val':True,'oracle_gate':False,
        'prior_closeout_used_internal_val_for_E4_eligibility':True,
        'token_accounting':'Exact tokenizer input_ids lengths, including chat template and special tokens; both passes counted for promotions',
        'fit_reference_baselines':base.groupby('k')[['f1','tokens','generation_latency']].mean().to_dict('index'),
        'source_hashes':sources}
    (output_dir/'frozen_controller.json').write_text(json.dumps(frozen,indent=2))
    pd.json_normalize(all_records).to_csv(output_dir/'fit_controller_grid.csv',index=False)
    print('Controller choices FROZEN on FIT:',frozen['selected_policy'],flush=True)
    # Confirmation only; never select or change thresholds from these results.
    val=apply_exact_token_costs(cached_split('internal_val'),'internal_val');qval=gate_inputs(val,'internal_val');vb=canonical_baselines(val)
    valprob=dict(zip(qval.question_id,gate.predict_proba(qval[U_COLUMNS].to_numpy())[:,1]))
    vpred=attach_saved_predictions(val,'internal_val')
    vscores=pd.read_csv(close/'internal_val_selector_scores.csv',dtype={'question_id':str,'paper_id':str})
    vscores=val[KEYS].merge(vscores,on=KEYS,validate='one_to_one',how='left')
    valmetrics={};fit_decisions={}
    for name,r in family_best.items():
        config=r['config'];selector=config['selector']
        vt=candidate_table(val,vscores[selector].to_numpy(),vpred['prob_help'],vpred['prob_hurt'])
        decisions=make_decisions(vt,valprob,config)
        metrics,realized=evaluate_decisions(decisions,val,vb);valmetrics[name]=metrics
        decisions.to_csv(output_dir/f'{name}_internal_val_decisions.csv',index=False)
        realized.to_csv(output_dir/f'{name}_internal_val_replay.csv',index=False)
        fd=make_decisions(tables[selector],fit_prob,config)
        _,fr=evaluate_decisions(fd,fit,base)
        fr.to_csv(output_dir/f'{name}_fit_oof_replay.csv',index=False)
    result={**frozen,'internal_val_metrics':valmetrics,
        'internal_val_reference_baselines':vb.groupby('k')[['f1','tokens','generation_latency']].mean().to_dict('index'),
        'evaluation':'saved-response replay; all observed harms retained; no oracle intervention',
        'internal_val_status':'Previously inspected development confirmation; no TEST used'}
    (output_dir/'controller_summary.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(valmetrics).T.to_csv(output_dir/'internal_val_metrics.csv')
    print(pd.DataFrame(valmetrics).T[['downstream_f1','promotion_rate','hurt_rate_all_queries','mean_cumulative_prompt_tokens']].to_string(),flush=True)
    return result
