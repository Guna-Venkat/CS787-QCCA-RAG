"""W8-H: real frozen-controller inference and matched single-promotion prompts.

Original SARA produces the shared draft. Actual generated token scores feed the
frozen B0-GW gate; actual MiniLM/U4-A inference feeds selection/safety. No outcome
column is admitted to the inference table. DAFiR-2 and automatic QCLA stacking
are unsupported because no corresponding frozen state-specific controller exists.
"""
from __future__ import annotations
from copy import deepcopy
import json,time
import joblib
import numpy as np
import pandas as pd
import torch
from src.week8.data import PROJECT_ROOT
from src.week8.generator_features import canonical_k4_context
from src.week8.gate_diagnostics import logits_diagnostics,summarize_diagnostics,U_COLUMNS
from src.week8.utility_training import load_selected_model,predict_neural,_validate_saved_routing
from src.week8.utility_dataset import assert_no_feature_leakage
from src.week8.semantic import SemanticScorer
from src.week8.marginal_information import compute_mig_features_for_query
from src.week8.selective_controller import make_decisions
from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt

IDENTITY=['question_id','paper_id','semantic_rank','candidate_passage_index']
SAFE_STATIC=IDENTITY+['question','candidate_text','bm25_rank','bm25_score','semantic_score','passage_length','base_context_tokens']


def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_continuation(model,tokenizer,prompt,embeddings,device,*,diagnostics=False):
    prefix=tokenizer(prompt,return_tensors='pt')['input_ids'].to(device)
    if embeddings.ndim!=2 or embeddings.shape[1]!=4096 or not torch.isfinite(embeddings).all():raise ValueError('Expected real flat compressed embeddings')
    synchronize(device);start=time.perf_counter()
    out=model.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),retrieval_embeds=embeddings.to(device=device,dtype=model.dtype),
        max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,output_scores=diagnostics)
    synchronize(device);seconds=time.perf_counter()-start
    # XMistral inputs_embeds generation returns continuation only, not prompt+answer.
    tokens=out.sequences[0].tolist()
    if len(tokens)>128:raise ValueError('SARA continuation contract changed; do not subtract prompt length')
    if tokenizer.eos_token_id in tokens:tokens=tokens[:tokens.index(tokenizer.eos_token_id)+1]
    result={'prediction':tokenizer.decode(tokens,skip_special_tokens=True).strip(),'generated_tokens':len(tokens),
        'token_ids':tokens,'prompt_tokens':prefix.shape[1],'generation_seconds':seconds}
    if diagnostics:
        if len(out.scores)!=out.sequences.shape[1] or not tokens:raise ValueError('Token/score alignment changed')
        begin=time.perf_counter()
        logits=torch.stack([s[0] for s in out.scores[:len(tokens)]])
        raw=logits_diagnostics(logits,tokens,tokenizer.eos_token_id)
        if not raw['greedy_verified']:raise ValueError('Generated IDs do not match greedy logits')
        result.update(diagnostics=raw,diagnostics_seconds=time.perf_counter()-begin)
    return result


class RevisionTokenizer:
    """Insert revision instructions inside the user turn before applying chat format."""
    def __init__(self,tokenizer,draft):self.tokenizer=tokenizer;self.draft=draft
    def __getattr__(self,name):return getattr(self.tokenizer,name)
    def apply_chat_template(self,messages,**kwargs):
        messages=deepcopy(messages)
        if messages[-1]['role']!='user':raise ValueError('Expected user context for revision')
        messages[-1]['content']+='\n\nInitial answer (JSON string): '+json.dumps(self.draft,ensure_ascii=False)+(
            '\nAdditional evidence has now been exposed. Revise the initial answer only if the evidence supports '
            'a correction or necessary additional detail. Retain supported correct details. Do not add unsupported '
            'information. Return only a succinct final answer.')
        return self.tokenizer.apply_chat_template(messages,**kwargs)


def promoted_prompt(record,group,embeddings,tokenizer,decision,mode,draft):
    if mode not in ['G0','G1']:raise ValueError('Only matched G0/G1 single promotion is implemented')
    if decision['selected_action']!='PROMOTE':raise ValueError('A STOP action must not build another generation prompt')
    _,_,_,heavy,_,full=canonical_k4_context(group,record,embeddings,tokenizer)
    selected=int(decision['candidate_passage_index'])
    candidate=group[group.candidate_passage_index.astype(int)==selected]
    if len(candidate)!=1:raise ValueError('Selected passage is not a canonical compressed candidate')
    row=candidate.iloc[0]
    for key in ['semantic_rank','bm25_rank']:
        if int(row[key])!=int(decision[key]):raise ValueError('Selected passage/rank identity changed')
    full=sorted(full+[selected]);heavy=sorted(set(range(10))-set(full))
    if mode=='G1' and not tokenizer.chat_template:raise ValueError('Draft-aware revision requires the canonical chat template')
    formatter=RevisionTokenizer(tokenizer,draft) if mode=='G1' else tokenizer
    prompt,_,compressed,_=build_mixed_fidelity_sara_prompt(str(record['question']).strip(),record['passages'][:10],full,[],heavy,{},formatter,hybrid_medium=False)
    if compressed!=heavy:raise ValueError('Compressed-slot order changed')
    return prompt,embeddings[compressed],full,compressed


class FrozenRuntimeController:
    def __init__(self,frozen):
        self.config=frozen['selected_config']
        if self.config['selector']!='S3' or self.config.get('uses_attribution') or frozen['max_promotions']!=1 or frozen['oracle_gate']:
            raise ValueError('This verified runtime supports the frozen S3 single-promotion controller only')
        self.gate=joblib.load(PROJECT_ROOT/'results/week8/w8e_controller/b0_gw_gate.joblib')
        name,self.selector,self.features,self.cfg=load_selected_model(PROJECT_ROOT/'results/week8/w8c_utility',device='cpu')
        if name!='U4-A':raise ValueError('Expected the real frozen U4-A safety model')
        assert_no_feature_leakage(self.features)
        self.scorer=SemanticScorer(device='cpu')

    def decide(self,record,routing,draft,raw_diagnostics):
        start=time.perf_counter()
        # Select columns before any features/model call. Outcomes and gold labels
        # cannot be accidentally consumed by inference or decisions.
        frame=routing[SAFE_STATIC].copy().sort_values('semantic_rank').reset_index(drop=True)
        frame['draft_answer']=draft
        passages=[p['text'] for p in record['passages'][:10]]
        sims=self.scorer.score_query_passages(record['question'],passages).cpu().numpy()
        full=_validate_saved_routing(frame,passages,sims)
        candidates={int(r.semantic_rank):str(r.candidate_text) for r in frame.itertuples()}
        feats=compute_mig_features_for_query(record['question'],draft,[passages[i] for i in full],candidates,self.scorer)
        for i,r in frame.iterrows():
            for key,value in feats[int(r.semantic_rank)].items():frame.loc[i,key]=value
        words=draft.lower().split();qwords=set(record['question'].lower().split());dwords=set(words)
        frame['draft_length_words']=len(words);frame['draft_empty']=int(not words)
        frame['draft_query_overlap']=len(qwords&dwords)/max(len(qwords|dwords),1)
        if not np.isfinite(frame[self.features].to_numpy(float)).all():raise ValueError('Missing real numeric selector features')
        question=str(record['question']).strip()
        semantic=float(self.scorer.score_query_passages(question,[draft]).cpu()[0]) if draft else 0.
        uncertainty=summarize_diagnostics(raw_diagnostics,question,draft,semantic)
        gate_probability=float(self.gate.predict_proba(np.asarray([[uncertainty[k] for k in U_COLUMNS]]))[:,1][0])
        predicted=predict_neural(self.selector,frame,self.cfg)
        table=frame[IDENTITY+['bm25_rank']].copy()
        table['selection_score']=frame.draft_relevance_ry.to_numpy()
        table['prob_help']=predicted['prob_help'];table['prob_hurt']=predicted['prob_hurt']
        qid=str(record['question_id'])
        action=make_decisions(table,{qid:gate_probability},self.config).iloc[0].to_dict()
        # Nullable integer STOP identities serialize as JSON null, never a fake index.
        for key in ['candidate_passage_index','semantic_rank','bm25_rank']:
            action[key]=None if pd.isna(action[key]) else int(action[key])
        for key in ['proposed_candidate_passage_index','proposed_semantic_rank','proposed_bm25_rank']:action[key]=int(action[key])
        return action,{'gate_features':uncertainty,'candidate_predictions':table.to_dict('records'),
            'decision_seconds':time.perf_counter()-start,'gate_probability':gate_probability}


def run_dafir_query(*args,**kwargs):
    """Reject the retired proxy-scoring API instead of silently running a mock system."""
    raise RuntimeError('Use run_w8h_full_evaluation.py and FrozenRuntimeController: real learned inference, canonical routing and matched G0/G1. Legacy proxy-scoring/multi-promotion execution is disabled.')
