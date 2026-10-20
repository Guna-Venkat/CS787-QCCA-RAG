"""Regression checks for paired identities and inference-only DEV routing."""
import numpy as np
import pandas as pd
import pytest
import torch
from src.week8.final_reporting import paired_interval
from src.week8.final_dev import dev_routing
from src.week8.generation import SAFE_STATIC


def test_paired_interval_rejects_partial_overlap_and_wrong_paper():
    base=pd.DataFrame({'question_id':['1','2'],'paper_id':['a','b'],'token_f1':[.2,.5]})
    with pytest.raises(ValueError):paired_interval(base.iloc[:1],base)
    wrong=base.copy();wrong.loc[0,'paper_id']='other'
    with pytest.raises(ValueError):paired_interval(wrong,base)
    with pytest.raises(ValueError):paired_interval(pd.concat([base,base]),base)


def test_bootstrap_identical_systems_and_constant_effect():
    base=pd.DataFrame({'question_id':['1','2','3'],'paper_id':['a','a','b'],'token_f1':[.2,.5,.1]})
    result=paired_interval(base,base,100)
    assert result['delta_f1']==result['ci95_low']==result['ci95_high']==0
    treatment=base.copy();treatment.token_f1+=.1
    result=paired_interval(treatment,base,100)
    assert np.allclose([result[k] for k in ['delta_f1','ci95_low','ci95_high']],.1)
    assert result['papers']==2 and result['queries']==3


class Scorer:
    def score_query_passages(self,*args):return torch.arange(10,dtype=torch.float32)/10


def test_dev_routing_has_real_candidate_identity_without_outcomes():
    record={'question_id':'1','paper_id':'p','question':'Q','gold_answers':['SECRET'],
        'passages':[{'text':str(i),'score':i/10} for i in range(10)]}
    ref={'selected_full_indices':[6,7,8,9],'prediction':'draft'}
    group=dev_routing(record,ref,Scorer())
    assert group.candidate_passage_index.tolist()==[5,4,3,2,1,0]
    assert group.semantic_rank.tolist()==list(range(5,11))
    assert not {'gold_answers','promoted_f1','base_f1','delta_f1'}&set(group)
    assert set(SAFE_STATIC)-set(group)=={'base_context_tokens'}
    ref['selected_full_indices']=[0,1,2,3]
    with pytest.raises(ValueError):dev_routing(record,ref,Scorer())


def test_dev_routing_retains_canonical_identity_for_ties():
    class TiedScorer:
        def score_query_passages(self,*args):return torch.ones(10)
    record={'question_id':'1','paper_id':'p','question':'Q','passages':[{'text':str(i)} for i in range(10)]}
    group=dev_routing(record,{'selected_full_indices':[6,7,8,9],'prediction':'draft'},TiedScorer())
    assert set(group.candidate_passage_index)==set(range(6))


def test_smoke_sample_is_fixed_and_depends_only_on_question_ids():
    from src.week8.final_dev import smoke_query_ids
    records={str(i):{'gold_answers':['secret']} for i in range(231)}
    chosen=smoke_query_ids(records)
    assert len(chosen)==len(set(chosen))==20
    assert chosen==smoke_query_ids({q:None for q in reversed(records)})
