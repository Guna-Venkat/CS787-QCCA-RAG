"""W8-H live inference, continuation accounting, matched prompts and no outcome inputs."""
from types import SimpleNamespace
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from src.week8.generation import generate_continuation,promoted_prompt,FrozenRuntimeController,run_dafir_query
from src.week8.full_system import evaluate_row,read_cache


class Tokenizer:
    eos_token_id=2;pad_token_id=0;chat_template='[INST]'
    def __call__(self,text,return_tensors=None):
        ids=list(range(10,19))
        return {'input_ids':torch.tensor([ids]) if return_tensors else ids}
    def apply_chat_template(self,messages,**kwargs):
        self.messages=deepcopy(messages)
        return '[INST] '+messages[-1]['content']+' [/INST]'
    def decode(self,ids,skip_special_tokens=True):return 'answer' if 5 in ids else ''


def fixture():
    record={'question_id':'q','paper_id':'p','question':' question ',
        'passages':[{'text':f'Passage {i}'} for i in range(10)],'gold_answers':['SECRET GOLD']}
    heavy=[0,2,4,5,6,8];sims=[.5,.9,.4,.8,.3,.2,.1,.7,0.,.6]
    rows=[{'question_id':'q','paper_id':'p','question':'question','candidate_passage_index':i,'semantic_rank':rank,
        'bm25_rank':i+1,'bm25_score':10.-i,'semantic_score':sims[i],'passage_length':2,
        'candidate_text':f'Passage {i}'} for rank,i in enumerate(heavy,5)]
    return record,pd.DataFrame(rows),torch.arange(10*4096,dtype=torch.float32).reshape(10,4096)


def test_continuation_ids_are_not_prompt_subtracted():
    class Generator:
        dtype=torch.float32
        def generate(self,**kwargs):
            assert kwargs['input_ids'].shape[1]==9
            assert kwargs['retrieval_embeds'].shape==(6,4096)
            scores=[]
            for token in [5,2]:
                x=torch.full((1,10),-2.);x[0,token]=3.;scores.append(x)
            return SimpleNamespace(sequences=torch.tensor([[5,2]]),scores=tuple(scores))
    result=generate_continuation(Generator(),Tokenizer(),'prompt',torch.ones(6,4096),'cpu',diagnostics=True)
    assert result['generated_tokens']==2 and result['prompt_tokens']==9
    assert result['prediction']=='answer' and result['diagnostics']['greedy_verified']
    assert result['diagnostics']['token_ids']==[5,2]


def test_g1_instruction_is_inside_user_turn_and_allocation_matches_g0():
    record,group,emb=fixture();tok=Tokenizer()
    action={'selected_action':'PROMOTE','candidate_passage_index':0,'semantic_rank':5,'bm25_rank':1}
    a=promoted_prompt(record,group,emb,tok,action,'G0','DRAFT TEXT')
    b=promoted_prompt(record,group,emb,tok,action,'G1','DRAFT TEXT')
    assert a[2:]==b[2:] and torch.equal(a[1],b[1])
    assert a[2]==[0,1,3,7,9] and b[1].shape==(5,4096)
    assert 'DRAFT TEXT' not in a[0]
    assert b[0].index('DRAFT TEXT')<b[0].index('[/INST]')
    assert tok.messages[-1]['role']=='user' and 'DRAFT TEXT' in tok.messages[-1]['content']
    assert 'SECRET GOLD' not in b[0]


def test_no_second_pass_on_stop_and_no_already_full_candidate():
    record,g,e=fixture()
    with pytest.raises(ValueError,match='STOP'):promoted_prompt(record,g,e,Tokenizer(),{'selected_action':'STOP'},'G1','draft')
    with pytest.raises(ValueError,match='canonical compressed'):promoted_prompt(record,g,e,Tokenizer(),{'selected_action':'PROMOTE','candidate_passage_index':1},'G0','draft')
    with pytest.raises(RuntimeError,match='Legacy proxy'):run_dafir_query(candidate_scorer_fn=lambda *args:[])


def test_live_decisions_do_not_read_outcome_columns(monkeypatch):
    import src.week8.generation as engine
    record,group,_=fixture();group['base_context_tokens']=900
    class Scorer:
        def score_query_passages(self,q,passages):
            return torch.tensor([.5,.9,.4,.8,.3,.2,.1,.7,0.,.6]) if len(passages)==10 else torch.tensor([.3])
    class Gate:
        def predict_proba(self,x):assert x.shape==(1,12);return np.array([[.1,.9]])
    controller=object.__new__(FrozenRuntimeController)
    controller.config={'family':'E5','policy_name':'E5-DraftSemantic','gate_threshold':.3,'tau_help':0.,'tau_hurt':.1}
    controller.gate=Gate();controller.scorer=Scorer();controller.selector=None;controller.features=['base_context_tokens'];controller.cfg=None
    monkeypatch.setattr(engine,'compute_mig_features_for_query',lambda *args:{r:{'draft_relevance_ry':11-r} for r in range(5,11)})
    def predict(model,frame,cfg):
        assert not set(frame)&{'gold_answers','delta_f1','promoted_f1','is_gold_evidence'}
        return {'prob_help':np.ones(6)*.2,'prob_hurt':np.ones(6)*.05}
    monkeypatch.setattr(engine,'predict_neural',predict)
    raw={'token_ids':[5,2],'token_logprobs':[-.2,-.1],'token_entropies':[1.,.5],'top1_top2_margins':[2.,3.],'eos_confidence':.9}
    a,_=controller.decide(record,group,'actual draft',raw)
    dirty=group.copy();dirty['delta_f1']=-100.;dirty['gold_answers']='secret';dirty['promoted_f1']=999.;dirty['is_gold_evidence']=True
    b,_=controller.decide(record,dirty,'actual draft',raw)
    assert a==b and a['selected_action']=='PROMOTE' and a['candidate_passage_index']==0


def test_outcome_harm_and_two_pass_costs_are_retained():
    decision={'selected_action':'PROMOTE','candidate_passage_index':0,'semantic_rank':5,'bm25_rank':1,'proposed_candidate_passage_index':0}
    base={'question_id':'q','paper_id':'p','decision':decision,'inference':{'gate_probability':.9,'decision_seconds':.1},
        'draft':{'prediction':'correct','prompt_tokens':90,'generated_tokens':2,'generation_seconds':1.,'diagnostics_seconds':.1},
        'base_pipeline_seconds':1.3,'reference_action_match':True,'full_indices':[1,3,7,9],'compressed_indices':[0,2,4,5,6,8]}
    revision={'prediction':'wrong','prompt_tokens':110,'generated_tokens':3,'generation_seconds':2.,'prompt_build_seconds':.01,
        'full_indices':[0,1,3,7,9],'compressed_indices':[2,4,5,6,8]}
    row=evaluate_row({'gold_answers':['correct']},base,revision,'original_SARA','G1','hash')
    assert row['token_f1']==0 and row['delta_f1']==-1 and row['prediction']=='wrong'
    assert row['cumulative_prompt_tokens']==200 and row['generated_tokens']==5 and row['num_generation_passes']==2
    base['decision']['selected_action']='STOP'
    for key in ['candidate_passage_index','semantic_rank','bm25_rank']:base['decision'][key]=None
    stopped=evaluate_row({'gold_answers':['correct']},base,None,'original_SARA','G1','hash')
    assert stopped['token_f1']==1 and stopped['cumulative_prompt_tokens']==90 and stopped['num_generation_passes']==1


def test_cache_rejects_stale_or_wrong_membership(tmp_path):
    path=tmp_path/'rows.jsonl';path.write_text(json.dumps({'fingerprint':'old','question_id':'q'})+'\n')
    with pytest.raises(ValueError,match='Stale'):read_cache(path,'new',{'q'},lambda r:r['question_id'])
    with pytest.raises(ValueError,match='extraneous'):read_cache(path,'old',{'different'},lambda r:r['question_id'])


def test_gpu_absence_fails_before_loading_models_or_running_data(tmp_path,monkeypatch):
    from src.week8.full_system import run_w8h
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='GPU-enabled sara_env'):run_w8h(tmp_path,smoke=True,device='cuda')


def test_completed_run_can_be_verified_without_cuda(tmp_path,monkeypatch):
    import src.week8.full_system as system
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(system,'frozen_policy',lambda:('frozen',None,tmp_path/'policy.csv'))
    monkeypatch.setattr(system,'load_g_variants',lambda *args:{})
    monkeypatch.setattr(system,'sources',lambda *args:('valid-fingerprint',{}))
    folder=tmp_path/'full';folder.mkdir()
    saved={'fingerprint':'valid-fingerprint','artifact_sha256':{},'completed':True}
    (folder/'summary.json').write_text(json.dumps(saved))
    assert system.run_w8h(tmp_path,device='cuda')==saved


def test_actual_tiny_xmistral_continuation_contract_on_cpu():
    # Real XMistral/Transformers generation, random tiny weights: this verifies
    # the API contract only and is explicitly not a real-checkpoint GPU smoke.
    from src.model.xMistral import XMistralConfig,XMistralForCausalLM
    cfg=XMistralConfig(vocab_size=128,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
        max_position_embeddings=512,retriever_hidden_size=4096,bos_token_id=1,eos_token_id=2,pad_token_id=0)
    model=XMistralForCausalLM(cfg).eval();model.compress_token_id=127
    class TinyTokenizer(Tokenizer):
        def __call__(self,text,return_tensors=None):return {'input_ids':torch.tensor([[1,20,127,127,127,127,127,127]])}
        def decode(self,ids,skip_special_tokens=True):return ' '.join(str(i) for i in ids if i not in [0,1,2,127])
    result=generate_continuation(model,TinyTokenizer(),'prompt',torch.randn(6,4096),'cpu',diagnostics=True)
    assert 1<=result['generated_tokens']<=128
    assert result['prompt_tokens']==8 and result['diagnostics']['greedy_verified']
    assert len(result['diagnostics']['token_ids'])==result['generated_tokens']
