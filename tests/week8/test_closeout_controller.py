"""Scientific invariants for cached closeout and non-oracle controller replay."""
import numpy as np
import pandas as pd
import pytest
import torch
from src.week8.closeout import heuristic_score,chosen_rows,bootstrap_difference
from src.week8.selective_controller import make_decisions,evaluate_decisions,choose_config
from src.week8.gate_diagnostics import logits_diagnostics,summarize_diagnostics


def candidates():
    return pd.DataFrame([dict(question_id='q',paper_id='p',semantic_rank=r,
        candidate_passage_index=i,bm25_rank=i+1,selection_score=s,prob_help=.6,prob_hurt=.1)
        for r,i,s in [(5,9,0),(6,2,1),(7,5,.2),(8,0,.1),(9,3,-.1),(10,7,-.2)]])


def test_controller_is_outcome_independent_and_preserves_identity():
    df=candidates();config=dict(policy_name='E3',family='E3',gate_threshold=.3)
    a=make_decisions(df,{'q':.5},config)
    dirty=df.assign(delta_f1=[1,-1,1,1,1,1],base_f1=1.,promoted_f1=0.,gold_answers='DO NOT READ',best_delta=1.)
    b=make_decisions(dirty,{'q':.5},config)
    pd.testing.assert_frame_equal(a,b)
    assert a.iloc[0].candidate_passage_index==2
    assert a.iloc[0].semantic_rank==6
    assert a.iloc[0].bm25_rank==3
    stopped=make_decisions(df,{'q':.1},config)
    assert stopped.iloc[0].selected_action=='STOP'
    assert pd.isna(stopped.iloc[0].candidate_passage_index)


def test_hurt_veto_uses_chosen_candidate_predictions_not_true_help():
    df=candidates();df.loc[df.semantic_rank==6,'prob_hurt']=.8
    cfg=dict(policy_name='E5',family='E5',gate_threshold=.2,tau_help=.1,tau_hurt=.2)
    a=make_decisions(df,{'q':.9},cfg)
    assert a.iloc[0].selected_action=='STOP'
    with pytest.raises(ValueError,match='Missing'):
        make_decisions(df,{},cfg)


def test_replay_keeps_harm_and_counts_both_generation_prompts():
    df=candidates();sf=df.drop(columns=['selection_score','prob_help','prob_hurt']).assign(
        delta_f1=[.3,-.2,0,0,0,0],promoted_f1=[.7,.2,.4,.4,.4,.4],
        base_f1=.4,base_context_tokens=100,promoted_context_tokens=150,promotion_latency=2.,best_delta=.3)
    baseline=pd.DataFrame([dict(question_id='q',k=4,f1=.4,tokens=100,generation_latency=1.),
                           dict(question_id='q',k=6,f1=.5,tokens=200,generation_latency=3.)])
    cfg=dict(policy_name='E3',family='E3',gate_threshold=.2)
    metrics,replay=evaluate_decisions(make_decisions(df,{'q':.9},cfg),sf,baseline)
    assert metrics['downstream_f1']==pytest.approx(.2) # oracle must not replace with .4
    assert metrics['hurt_rate_all_queries']==1
    assert metrics['mean_cumulative_prompt_tokens']==250
    assert metrics['estimated_generation_latency_sec']==3
    stopped,_=evaluate_decisions(make_decisions(df,{'q':.1},cfg),sf,baseline)
    assert stopped['downstream_f1']==pytest.approx(.4)
    assert stopped['mean_cumulative_prompt_tokens']==100


def test_config_selection_uses_point003_cost_latency_then_simplicity():
    def record(f1,cost,latency,complexity):
        return dict(config=dict(complexity=complexity,policy_name=str(complexity)),
            metrics=dict(downstream_f1=f1,mean_cumulative_prompt_tokens=cost,estimated_generation_latency_sec=latency))
    best=record(.42,200,2,2);cheap=record(.418,150,3,3);too_low=record(.416,50,1,0)
    assert choose_config([best,cheap,too_low]) is cheap
    fast=record(.418,150,2,5)
    assert choose_config([best,cheap,fast]) is fast
    simple=record(.419,150,2,1)
    assert choose_config([fast,simple]) is simple


def test_logits_diagnostics_uses_actual_selected_token_and_float32_entropy():
    logits=torch.tensor([[4.,1.,0.],[0.,1.,4.]],dtype=torch.bfloat16)
    raw=logits_diagnostics(logits,[0,2],2)
    assert raw['greedy_verified']
    expected=torch.log_softmax(logits.float(),dim=-1)
    assert raw['token_logprobs']==pytest.approx([float(expected[0,0]),float(expected[1,2])])
    assert not logits_diagnostics(logits,[1,2],2)['greedy_verified']
    f=summarize_diagnostics(raw,'question','answer',.2)
    assert f['u6_seq_logprob']==pytest.approx(sum(raw['token_logprobs']))
    assert f['u5_logprob_mean']==pytest.approx(np.mean(raw['token_logprobs']))
    assert f['u12_semantic_sim']==.2


def test_cached_selectors_ties_missingness_and_paired_bootstrap():
    df=candidates().assign(draft_relevance_ry=.2,mig_score=.1,
        masking_delta_logp_per_token=np.nan,masking_available=0,
        base_f1=.4,promoted_f1=.5,delta_f1=.1,best_delta=.1,
        base_context_tokens=100,promoted_context_tokens=150)
    assert np.array_equal(heuristic_score(df,'S5'),heuristic_score(df,'S3'))
    assert chosen_rows(df.iloc[::-1],np.zeros(6)).iloc[0].semantic_rank==5
    b=bootstrap_difference(df,np.zeros(6),np.zeros(6),n=10)
    assert b['f1']=={'difference':0.,'ci95':[0.,0.]}


def test_batched_generation_keeps_actual_tokens_and_ignores_padding_after_eos():
    from types import SimpleNamespace
    from src.week8.gate_diagnostics import measure_draft_batch
    class Tok:
        pad_token_id=3
        eos_token_id=2
        def __call__(self,text,return_tensors='pt'):
            return {'input_ids':torch.tensor([[7,8,9]] if text=='long prompt' else [[8,9]])}
        def decode(self,ids,skip_special_tokens=True):
            return 'a' if ids==[0,2] else 'bb' if ids==[1,1,2] else 'unexpected'
    class Model:
        dtype=torch.float32
        def generate(self,**kwargs):
            assert kwargs['attention_mask'].tolist()==[[1,1,1],[0,1,1]]
            assert kwargs['retrieval_embeds'].shape==(12,2)
            scores=[]
            for targets in [(0,1),(2,1),(3,2)]:
                logits=torch.zeros(2,4)
                for i,target in enumerate(targets):logits[i,target]=5
                scores.append(logits)
            return SimpleNamespace(sequences=torch.tensor([[0,2,3],[1,1,2]]),scores=scores)
    items=[dict(question_id='q1',prompt='long prompt',draft_answer='a',embeds=torch.ones(6,2)),
           dict(question_id='q2',prompt='short prompt',draft_answer='bb',embeds=torch.ones(6,2))]
    result=measure_draft_batch(Model(),Tok(),items,'cpu')
    assert result['q1']['token_ids']==[0,2]
    assert len(result['q1']['token_logprobs'])==2
    assert result['q2']['token_ids']==[1,1,2]
    assert result['q1']['greedy_verified'] and result['q2']['greedy_verified']


def test_quality_tolerance_is_not_applied_twice_across_families():
    from src.week8.selective_controller import choose_family_operating_points
    def r(name,f1,cost):
        return dict(config=dict(policy_name=name,complexity=1),metrics=dict(
            downstream_f1=f1,mean_cumulative_prompt_tokens=cost,estimated_generation_latency_sec=1.))
    a_best=r('A',.420,500);a_cheap=r('A',.4175,300)
    b_best=r('B',.418,200);b_cheap=r('B',.4155,100)
    selected,representatives=choose_family_operating_points({'A':[a_best,a_cheap],'B':[b_best,b_cheap]})
    assert selected is b_best
    assert .420-selected['metrics']['downstream_f1']<=.003
    assert representatives['B'] is b_best


def test_attribution_cost_is_counted_once_per_promoted_query():
    df=candidates();sf=df.drop(columns=['selection_score','prob_help','prob_hurt']).assign(
        delta_f1=.1,promoted_f1=.5,base_f1=.4,base_context_tokens=100,
        promoted_context_tokens=150,promotion_latency=2.,best_delta=.1,feature_extraction_seconds_query=5.)
    baseline=pd.DataFrame([dict(question_id='q',k=4,f1=.4,tokens=100,generation_latency=1.),
                           dict(question_id='q',k=6,f1=.5,tokens=200,generation_latency=3.)])
    cfg=dict(policy_name='E4',family='E4',gate_threshold=.2,uses_attribution=True)
    metrics,_=evaluate_decisions(make_decisions(df,{'q':.9},cfg),sf,baseline)
    assert metrics['estimated_attribution_latency_sec']==5.
    assert metrics['estimated_total_recorded_latency_sec']==8.
    stopped,_=evaluate_decisions(make_decisions(df,{'q':.1},cfg),sf,baseline)
    assert stopped['estimated_attribution_latency_sec']==0.
    with pytest.raises(ValueError,match='attribution cost'):
        evaluate_decisions(make_decisions(df,{'q':.9},cfg),sf.drop(columns='feature_extraction_seconds_query'),baseline)


def test_exact_cost_accounting_does_not_redefine_model_features():
    df=candidates();sf=df.drop(columns=['selection_score','prob_help','prob_hurt']).assign(
        delta_f1=.1,promoted_f1=.5,base_f1=.4,base_context_tokens=100,
        promoted_context_tokens=150,promotion_latency=2.,best_delta=.1,
        exact_base_prompt_tokens=108,exact_promoted_prompt_tokens=158)
    baseline=pd.DataFrame([dict(question_id='q',k=4,f1=.4,tokens=108,generation_latency=1.),
                           dict(question_id='q',k=6,f1=.5,tokens=208,generation_latency=3.)])
    cfg=dict(policy_name='E1',family='E1')
    metrics,replay=evaluate_decisions(make_decisions(df,{'q':.9},cfg),sf,baseline)
    assert metrics['mean_cumulative_prompt_tokens']==266
    assert metrics['mean_final_prompt_tokens']==158
    assert sf.base_context_tokens.eq(100).all()
    assert replay.base_context_tokens.eq(100).all()
    with pytest.raises(ValueError,match='Incomplete exact'):
        evaluate_decisions(make_decisions(df,{'q':.9},cfg),sf.drop(columns='exact_base_prompt_tokens'),baseline)
