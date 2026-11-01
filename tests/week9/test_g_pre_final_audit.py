"""Post-result audit tests; no real G learner fits or held-out data evaluation."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.week9.provenance import ROOT,branch_guard,digest,file_hash
from src.week9.action_verification_g import features,controller
from src.week9.action_verification_g_audit import audit,final_fit

AUDIT=ROOT/'results/week9/w9g_action_verification/v1/audit_pre_final'


@pytest.fixture(scope='module')
def case():return audit.load_case()


def test_exact_query_paper_counts_and_unique_policy_rows(case):
    assert len(case['meta'])==1746 and case['meta'].paper_id.nunique()==656
    rows=audit.csv_rows(case['oof']/'oof_predictions.csv')
    pairs=[(r['question_id'],r['model']) for r in rows]
    assert len(pairs)==len(set(pairs))==3492


def test_specific_action_target_and_canonical_hashes(case):
    rows=audit.audit_lineage(case)
    assert len(rows)==1746 and sum(r['accept_this_f0'] for r in rows)==378
    assert sum(r['state']=='HURT' for r in rows)==365
    assert sum(r['state']=='NEUTRAL' for r in rows)==1003
    assert all(len(r['d0_answer_hash'])==len(r['f0_answer_hash'])==len(r['evidence_hash'])==64 for r in rows)


@pytest.mark.parametrize('forbidden',['gold_answer','f1','delta_f1','HELP','HURT','accept_label','oracle_action',
    'paper_id','question_id','row_order','selected_answer','threshold_result','fold_statistics'])
def test_all_forbidden_matrix_columns_rejected(case,forbidden):
    record=case['x'].iloc[0].to_dict();record[forbidden]=1
    with pytest.raises(ValueError):features.feature_matrix([record],case['cfg']['feature_groups'])


def test_exact_frozen_schema_and_no_outcome_dependency(case):
    assert case['x'].shape==(1746,32) and list(case['x'])==case['p']['features']
    q=case['meta'].iloc[0].question_id;g=case['bundle']['groups'];src=case['bundle']['e3'][q]
    before=features.online_features(features.payload(g['D0'][q],g['F0'][q],src))
    base=dict(g['D0'][q],f1=999,gold_answer='DO_NOT_USE',oracle_action='ACCEPT')
    fresh=dict(g['F0'][q],f1=-999,delta_f1=999,help=True,hurt=True)
    after=features.online_features(features.payload(base,fresh,src))
    assert before==after


def test_row_order_not_a_feature(case):
    bundle=dict(case['bundle']);bundle['groups']=dict(bundle['groups'])
    bundle['groups']['D0']=dict(reversed(list(bundle['groups']['D0'].items())))
    x,meta=features.prepare(bundle,case['cfg'])
    assert meta.equals(case['meta']) and x.equals(case['x'])


def test_no_global_fitted_preprocessing_in_feature_builder():
    tree=ast.parse(Path(features.__file__).read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','fit_transform') for n in ast.walk(tree))
    trace=json.loads((AUDIT/'preprocessing_call_trace.json').read_text())
    assert trace['real_classifier_fits']==0 and trace['fresh_pipeline_instances']==52
    assert trace['all_preprocessing_training_only'] and trace['no_prediction_on_fit_rows']
    assert trace['full_fit_estimators_never_used_for_oof_predictions']


def test_outer_and_inner_paper_overlaps_all_zero(case):
    _,splits,rows=audit.audit_folds(case)
    assert len(rows)==40 and len(splits)==5
    for row in rows:
        assert all(v==0 for k,v in row.items() if k.endswith('_overlap'))


def test_inner_threshold_selection_cannot_read_outer_labels():
    trace=json.loads((AUDIT/'preprocessing_call_trace.json').read_text())
    assert trace['outer_heldout_label_reads_during_inner_threshold_selection']==0
    calls=trace['threshold_calls']
    assert len(calls)==12 and sum(r['pooled_deployment_selection'] for r in calls)==2
    assert all(r['outside_allowed_label_reads']==0 for r in calls)


def test_threshold_rules_and_oof_decision_reconstruction(case):
    _,splits,_=audit.audit_folds(case)
    _,rows,coverage=audit.audit_thresholds(case,splits)
    log=[r for r in rows if r['model']=='logistic' and r['outer_fold']>=0]
    tree=[r for r in rows if r['model']=='hist_gradient_boosting' and r['outer_fold']>=0]
    assert [r['selected_threshold'] for r in log]==[.7,.7,.7,1.,.7]
    assert [r['selected_threshold'] for r in tree]==[.4,.4,.4,.8,.3]
    assert all(r['expected_predictions']==r['actual_predictions'] for r in coverage)


def test_threshold_ties_are_deterministic():
    rows=[dict(f1=.5,hurt=.1,acceptance_rate=.2,total_query_prompt_tokens=100,threshold=t) for t in [.4,.6]]
    # Use exactly symmetric representable tie values to check final ordering.
    rows=[dict(r,threshold=t) for r,t in zip(rows,[.25,.75])]
    chosen,best,eligible=audit.pick_threshold(rows)
    assert chosen['threshold']==.75 and best==.5 and len(eligible)==2
    assert audit.pick_threshold(list(reversed(rows)))[0]['threshold']==.75


def test_pooled_deployment_not_mislabeled_nested_oof(case):
    t=audit.csv_rows(case['oof']/'threshold_selection.csv')
    pooled=[r for r in t if r['model']=='logistic' and int(r['outer_fold'])==-1]
    selected,_,_=audit.pick_threshold(pooled)
    summary=next(r for r in audit.csv_rows(case['oof']/'summary.csv') if r['model']=='logistic')
    assert float(selected['threshold'])==.7
    assert not np.isclose(float(selected['f1']),float(summary['f1']))
    assert summary['evaluation']=='nested-paper-OOF'


def test_saved_hyperparameters_and_fit_schema_frozen(case):
    data=audit.audit_models(case)
    log=data['logistic'];tree=data['hist_gradient_boosting']
    assert log['params']['C']==1 and log['params']['class_weight']=='balanced' and log['standardized']
    assert log['params']['random_state']==42 and log['training_rows_evidence']==1746
    assert tree['params']['max_iter']==100 and tree['params']['max_leaf_nodes']==7
    assert tree['params']['max_depth']==3 and tree['params']['min_samples_leaf']==20
    assert tree['params']['l2_regularization']==1 and tree['params']['random_state']==42
    assert tree['params']['class_weight'] is None
    assert log['sklearn']==case['p']['sklearn']


def test_actual_missingness_is_zero_and_missing_measurements_not_zero(case):
    assert case['x'].isna().sum().sum()==0
    q=case['meta'].iloc[0].question_id;g=case['bundle']['groups']
    payload=features.payload(g['D0'][q],g['F0'][q],case['bundle']['e3'][q])
    payload['ranking']['next_unselected_cosine']=None
    payload['generation_lengths']['candidate_generated_tokens']=None
    f=features.online_features(payload)
    assert np.isnan(f['next_unselected_cosine']) and np.isnan(f['candidate_generated_tokens'])


def test_selected_answers_help_hurt_and_equal_weight_metrics(case):
    rows=audit.csv_rows(case['oof']/'oof_predictions.csv')
    measured,_=audit.independently_measure(case,rows)
    log=next(r for r in measured if r['model']=='logistic')
    assert np.isclose(log['f1'],.3894432529533657)
    assert log['accept_on_help']==130 and log['accept_on_hurt']==109 and log['keep_on_hurt']==256
    assert log['accepted_neutral_count']==82
    assert np.isclose(log['harm_rejection_rate'],256/365) and np.isclose(log['help_retention_rate'],130/378)
    assert np.isclose(log['acceptance_rate'],321/1746) and np.isclose(log['false_accept_rate'],191/321)


def test_post_generation_keep_does_not_refund_candidate(case):
    rows=audit.csv_rows(case['oof']/'oof_predictions.csv')
    kept=[r for r in rows if r['model']=='logistic' and r['accepted']=='False']
    assert kept
    for r in kept:
        f0=case['bundle']['groups']['F0'][r['question_id']]
        assert float(r['num_lm_calls'])==2
        assert float(r['total_query_prompt_tokens'])==f0['total_query_prompt_tokens']
        assert r['measured_end_to_end_latency']==''


def test_independent_cluster_bootstrap_matches_saved(case):
    rows=audit.csv_rows(case['oof']/'oof_predictions.csv')
    log={r['question_id']:r for r in rows if r['model']=='logistic'}
    result=audit.bootstrap(log,case['bundle']['groups']['D0'])
    assert result['papers']==656 and result['resamples']==10000 and result['seed']==42
    assert np.isclose(result['point_estimate'],.0038120764555066005)
    assert np.allclose([result['ci95_low'],result['ci95_high']],[-.005019389471817261,.012492006908273522])


def test_bootstrap_clusters_entire_papers():
    a={str(i):dict(paper_id='same',f1=float(i%2)) for i in range(4)}
    b={str(i):dict(paper_id='same',f1=0.) for i in range(4)}
    result=audit.bootstrap(a,b,n=1000)
    assert result['ci95_low']==result['ci95_high']==.5


def test_final_fit_cannot_overwrite_oof(tmp_path):
    root=tmp_path/'g';root.mkdir()
    with pytest.raises(AssertionError):final_fit.validate_destination(root,root/'controller_fit_oof')
    assert final_fit.validate_destination(root,root/'final_fit')==root/'final_fit'
    (root/'final_fit').mkdir()
    with pytest.raises(AssertionError):final_fit.validate_destination(root,root/'final_fit')


def test_final_fit_requires_exact_original_schema(case):
    status=json.loads((AUDIT/'audit_status.json').read_text())
    schema=final_fit.frozen_schema(case['x'].columns,status,case['p'])
    assert schema['feature_count']==32
    with pytest.raises(AssertionError):final_fit.frozen_schema(list(reversed(case['x'].columns)),status,case['p'])
    with pytest.raises(AssertionError):final_fit.frozen_schema(list(case['x'].columns)+['new_semantic_feature'],status,case['p'])


def test_invalid_audit_blocks_final_fit_before_any_fit(tmp_path):
    path=tmp_path/'invalid.json';path.write_text(json.dumps(dict(status='INVALID — RERUN REQUIRED',safe_for_final_fit=False,oof_rerun_required=True)))
    with pytest.raises(AssertionError,match='valid matching audit'):final_fit.validate_audit(path)


def test_valid_audit_gate_without_running_fit():
    status=final_fit.validate_audit(AUDIT/'audit_status.json')
    assert status['safe_for_final_fit'] and not status['oof_rerun_required']


def test_changed_input_blocks_final_fit(tmp_path):
    status=json.loads((AUDIT/'audit_status.json').read_text())
    value=tmp_path/'input.json';value.write_text('original')
    status['input_hashes']={str(value):file_hash(value)}
    path=tmp_path/'audit.json';path.write_text(json.dumps(status))
    value.write_text('changed')
    with pytest.raises(ValueError,match='changed'):final_fit.validate_audit(path)


def test_final_fit_requires_explicit_action_and_no_evaluation_path():
    with pytest.raises(SystemExit):final_fit.main(['--audit','fixture'])
    tree=ast.parse(Path(final_fit.__file__).read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('predict','predict_proba','score') for n in ast.walk(tree))


def test_branch_gagan_only():assert branch_guard()=='gagan'


def test_original_artifact_hashes_unchanged(case):
    for path,expected in case['protected'].items():assert file_hash(path)==expected,path
