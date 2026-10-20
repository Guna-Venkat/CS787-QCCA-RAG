"""Scientific and runtime invariants for repaired W8-G."""
from copy import deepcopy
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
from src.week8.hard_case_dataset import build_hard_case_dataset,HardCasePromptDataset,compute_outcome_weights
from src.week8.hard_case_lora import DeltaLoRALinear,setup_lora_generator,adapter_state,load_adapter_state
from src.week8.outcome_weighted_training import evaluation_context,summarize


class Tokenizer:
    eos_token_id=9
    chat_template="test"
    def encode(self,text,add_special_tokens=False):return [1,2,3]
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return 'CHAT '+messages[-1]['content']+' ASSISTANT'
    def __call__(self,text,return_tensors=None,add_special_tokens=True):
        ids=[1,2,3] if add_special_tokens else [4,5]
        return {'input_ids':torch.tensor([ids]) if return_tensors else ids}


def fixture(delta=.08):
    record={'question_id':'q','paper_id':'p','question':' question ',
        'passages':[{'text':f'Passage {i}'} for i in range(10)],'gold_answers':['true gold']}
    # Original BM25 index 0 is semantic rank 5, and must be eligible for promotion.
    heavy=[0,2,4,5,6,8]
    rows=[{'question_id':'q','paper_id':'p','question':'question','semantic_rank':rank,
        'bm25_rank':i+1,'candidate_passage_index':i,'candidate_text':f'Passage {i}',
        'delta_f1':delta if i==0 else 0.,'draft_answer':'baseline draft'} for rank,i in enumerate(heavy,5)]
    return record,rows


def test_training_uses_semantic_identity_and_original_slots():
    record,rows=fixture()
    examples,counts=build_hard_case_dataset([record],rows,Tokenizer(),mode='outcome_weighted',gamma=2)
    ex=examples[0]
    assert ex['baseline_full_indices']==[1,3,7,9]
    assert ex['full_indices']==[0,1,3,7,9]
    assert ex['compressed_indices']==[2,4,5,6,8]
    assert ex['training_promoted_passage_index']==0 and ex['weight']==pytest.approx(1.16)
    assert counts['TYPE_A_HELPFUL']==1


@pytest.mark.parametrize('delta,kind,weight',[(-.2,'TYPE_B_DISTRACTOR_SAFE',1.4),(0.,'TYPE_C_NEUTRAL',1.)])
def test_nonhelpful_training_keeps_canonical_k4(delta,kind,weight):
    record,rows=fixture(delta)
    examples,_=build_hard_case_dataset([record],rows,Tokenizer(),mode='outcome_weighted',gamma=2)
    e=examples[0]
    assert e['full_indices']==[1,3,7,9] and len(e['compressed_indices'])==6
    assert e['case_type']==kind and e['weight']==pytest.approx(weight)


def test_missing_and_wrong_split_promotions_fail():
    record,rows=fixture()
    with pytest.raises(ValueError,match='six canonical'):build_hard_case_dataset([record],rows[:1],Tokenizer())
    wrong=deepcopy(rows)
    for row in wrong:row['question_id']='validation_query'
    with pytest.raises(ValueError,match='coverage'):build_hard_case_dataset([record],wrong,Tokenizer())
    with pytest.raises(ValueError,match='Missing FIT'):build_hard_case_dataset([record],[],Tokenizer())


def test_labels_preserve_generation_prefix_and_eos():
    record,rows=fixture()
    ex,_=build_hard_case_dataset([record],rows,Tokenizer())
    item=HardCasePromptDataset(ex,Tokenizer())[0]
    assert item['input_ids'].tolist()==[[1,2,3,4,5,9]]
    assert item['targets'].tolist()==[[4,5,9]]
    with pytest.raises(ValueError,match='never truncate'):HardCasePromptDataset(ex,Tokenizer(),max_seq_len=4)[0]


def test_fixed_evaluation_ignores_outcomes_and_keeps_identity():
    record,rows=fixture();g=pd.DataFrame(rows);emb=torch.randn(10,4096)
    action=pd.Series({'paper_id':'p','selected_action':'PROMOTE','candidate_passage_index':0,'semantic_rank':5,'bm25_rank':1})
    a=evaluation_context(record,g,emb,Tokenizer(),action)
    dirty=g.copy();dirty['delta_f1']=-100.;dirty['promoted_f1']=999.;dirty['is_gold_evidence']=True
    b=evaluation_context(record,dirty,emb,Tokenizer(),action)
    assert a[0]==b[0] and torch.equal(a[1],b[1]) and a[2:]==b[2:]
    assert a[2]==[0,1,3,7,9] and a[4]==0
    stop=action.copy();stop['selected_action']='STOP'
    for k in ['candidate_passage_index','semantic_rank','bm25_rank']:stop[k]=pd.NA
    c=evaluation_context(record,dirty,emb,Tokenizer(),stop)
    assert c[2]==[1,3,7,9] and c[-1]=='STOP'


def test_new_lora_preserves_original_projection_and_trains_only_delta():
    original=torch.nn.Linear(4,3)
    before={n:p.detach().clone() for n,p in original.named_parameters()}
    layer=DeltaLoRALinear(original,r=2,alpha=4,dropout=0.)
    x=torch.randn(2,4)
    assert torch.equal(layer(x),original(x))
    optimizer=torch.optim.SGD([p for p in layer.parameters() if p.requires_grad],lr=.1)
    layer(x).square().mean().backward();optimizer.step()
    assert layer.w8g_B.abs().sum()>0
    for n,p in original.named_parameters():assert p.grad is None and torch.equal(p,before[n])


def tiny_model():
    model=torch.nn.Module();model.layer=torch.nn.Module();model.layer.self_attn=torch.nn.Module()
    for name in ['q_proj','k_proj','v_proj','o_proj']:setattr(model.layer.self_attn,name,torch.nn.Linear(4,4))
    return model


def test_independent_adapter_checkpoint_and_no_double_wrap():
    model=setup_lora_generator(tiny_model(),r=2)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name.endswith('w8g_B'):p.fill_(.02)
    saved=adapter_state(model)
    fresh=setup_lora_generator(tiny_model(),r=2);load_adapter_state(fresh,saved)
    assert all(torch.equal(adapter_state(fresh)[k],v) for k,v in saved.items())
    assert all(n.endswith(('w8g_A','w8g_B')) for n,p in fresh.named_parameters() if p.requires_grad)
    with pytest.raises(ValueError,match='already attached'):setup_lora_generator(model)


def test_harm_is_retained_in_reported_downstream_f1():
    common={'question_id':'q','context':'frozen_controller','selected_action':'PROMOTE',
        'final_prompt_tokens':100,'cumulative_prompt_tokens':180,'estimated_cumulative_generation_latency_sec':3.}
    result=summarize([{**common,'token_f1':.1}],[{**common,'token_f1':.5}])['frozen_controller']
    assert result['downstream_f1']==.1 and result['gain_over_original_same_policy']==pytest.approx(-.4)
    assert result['hurt_rate_vs_original']==1 and result['mean_cumulative_prompt_tokens']==180


def test_all_variants_use_identical_training_contexts():
    r,rows=fixture()
    examples=[build_hard_case_dataset([r],rows,Tokenizer(),mode=m)[0][0] for m in ['uniform','hard_case','outcome_weighted']]
    assert len({e['prompt_text'] for e in examples})==1
    assert len({e['weight'] for e in examples})==3
    assert compute_outcome_weights(-.2)==1.


def test_weighted_accumulation_and_remainder_use_query_mean(monkeypatch):
    import src.week8.hard_case_lora as training
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w8g_B=torch.nn.Parameter(torch.tensor(0.))
            self.original=torch.nn.Parameter(torch.tensor(3.),requires_grad=False)
    m=Model();optimizer=torch.optim.SGD([m.w8g_B],lr=.1)
    monkeypatch.setattr(training,'training_mode',lambda model:None)
    monkeypatch.setattr(training,'compressed_embeddings',lambda *args:None)
    monkeypatch.setattr(training,'answer_loss',lambda model,ids,target,emb:(model.w8g_B-target).square().mean())
    def batch(target,weight):return {'input_ids':torch.zeros(1,1,dtype=torch.long),'targets':torch.tensor([[target]]),'weight':weight}
    batches=[batch(.1,.5) for _ in range(4)]+[batch(.03,2.)]
    result=training.train_generator_lora_epoch(m,batches,optimizer,{},device='cpu',gradient_accumulation_steps=4)
    # Four half-weight queries update 0 -> .01; the partial group updates .01 -> .018.
    assert float(m.w8g_B.detach())==pytest.approx(.018,abs=1e-6)
    assert result['query_mean_weighted_nll']==pytest.approx((4*.005+2*.02**2)/5,abs=1e-6)
    assert result['query_mean_unweighted_nll']==pytest.approx((4*.01+.02**2)/5,abs=1e-6)
    assert m.original.grad is None and float(m.original)==3.
