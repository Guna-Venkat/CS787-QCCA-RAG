"""Fabricated all-adapter exercise. Hardware/learned predictions are explicit doubles.

No fixture text or identity is taken from QASPER; this is software validation,
not scientific performance evidence. Actual prompt, selection, source validation,
cost, frozen W8 serialization/diagnostics and G2 feature code is exercised.
"""
import contextlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch
from .producer import World,ORDER,dispatch
from . import adapters
from .history import historical
from .storage import pair_membership
from ..final_h import protocol as p


class ToyScorer:
    def score_query_passages(self,query,passages):return torch.tensor([1-i/20 for i in range(len(passages))])


class ToyEncoder:
    def encode(self,texts):
        return np.asarray([np.random.default_rng(int(p.digest(text)[:8],16)).normal(size=8) for text in texts])


class ToyBackend:
    device='cpu';model=SimpleNamespace(dtype=torch.float32);answer_max_tokens=128
    def __init__(self,tokenizer,identity):self.tokenizer=tokenizer;self._identity=identity;self.invocations=[]
    def identity(self):return self._identity
    def render(self,messages):return self.tokenizer.apply_chat_template([dict(role='user',content='\n\n'.join(m['content'] for m in messages))],tokenize=False,add_generation_prompt=True)
    def generate(self,prompt,max_tokens,retrieval_embeds=None):
        self.invocations.append(max_tokens)
        text='candidate synthetic answer' if 'Additional evidence:' in prompt else 'initial synthetic answer'
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        return dict(content=text,prompt_tokens=len(self.tokenizer.encode(prompt)),generated_tokens=len(ids),
                    generated_token_ids=ids,latency=.001,**self.identity())


class ToyPipeline:
    def predict_proba(self,x):
        prob=np.where(x.query_tokens.to_numpy()>4,.9,.1)
        return np.column_stack((1-prob,prob))


class ToyW8Model:
    dtype=torch.float32
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    def generate(self,**kwargs):
        ids=self.tokenizer.encode('initial synthetic answer',add_special_tokens=False)
        scores=[]
        for token in ids:
            logits=torch.zeros((1,len(self.tokenizer)));logits[0,token]=10;scores.append(logits)
        return SimpleNamespace(sequences=torch.tensor([ids]),scores=scores)


class ToyController:
    def decide(self,record,group,draft,diagnostics):
        from src.week8.selective_controller import make_decisions
        frame=group[['question_id','paper_id','semantic_rank','candidate_passage_index','bm25_rank']].copy()
        frame['selection_score']=1/frame.semantic_rank;frame['prob_help']=.8;frame['prob_hurt']=0.
        probability=1. if str(record['question_id']).endswith('1') else 0.
        action=make_decisions(frame,{str(record['question_id']):probability},p.read(p.ROOT/'results/week8/w8e_controller/frozen_controller.json')['selected_config']).iloc[0].to_dict()
        for name in ('candidate_passage_index','semantic_rank','bm25_rank'):
            action[name]=None if pd.isna(action[name]) else int(action[name])
        return action,dict(gate_probability=probability,decision_seconds=.001)


def run(root):
    from ..backend import load_tokenizer
    from ..final_h.analysis import evaluate,report
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cfg=dict(checkpoint='Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42',query_draft_alpha=.5)
    tokenizer=load_tokenizer(cfg)
    model=p.read(p.F_ROOT/'frozen_w9f_intervention.json')['generator_identity']
    freeze=dict(manifest_digest='SYNTHETIC_ONLY',model=model,encoder=p.read(p.E3_ROOT/'protocol.json')['encoder'],success_criteria=p.success_criteria())
    records=[]
    for i,question in enumerate(('What synthetic result?','What is the synthetic result?')):
        passages=[dict(passage_id=f'SYNTHETIC-p{i}-d{j}',rank=j+1,score=float(10-j),text=f'Fabricated passage {j}. Synthetic measurement is blue. The toy result is explicit.') for j in range(10)]
        records.append(dict(question_id=f'SYNTHETIC-q{i}',paper_id=f'SYNTHETIC-p{i}',question=question,passages=passages,
            source_context=[x['text'] for x in passages],gold_answers=['synthetic answer'],gold_answers_reformatted=['synthetic answer']))
    saved={};toy=ToyBackend(tokenizer,model);world=World.__new__(World)
    world.root=root;world.freeze=freeze;world.cfg=cfg;world._encoder=ToyEncoder();world.backend=toy
    world.embeddings={r['question_id']:torch.zeros((10,4096)) for r in records};world._calls=None;world._ext=None;world._sentence=None
    world.static_scorer=lambda:ToyScorer();world.scorer=lambda:ToyScorer();world.generator=lambda:toy
    world.saved=lambda s,q:saved[s,q]
    world.verifier=lambda:(ToyPipeline(),p.read(p.G_ROOT/'final_fit/feature_schema.json')['feature_columns'])
    def pool(record,d0):
        inf=record['question_id'];candidates=[]
        for j,passage in enumerate(record['passages'][:4]):
            candidates.append(dict(chunk_id=f'SYNTHETIC-chunk-{inf}-{j}',paper_id=record['paper_id'],text=passage['text'],
                char_start=j*200,char_end=j*200+len(passage['text']),token_length=len(tokenizer.encode(passage['text'],add_special_tokens=False))))
        return dict(candidates=candidates,retrieval_query_time=.001,retrieval_width=20,retrieval_actual_count=4)
    world.pool=pool
    with contextlib.ExitStack() as stack:
        world.stack=stack
        for system in ORDER:
            world.system=system;world.identity=dict(synthetic=True,system=system);world._calls=None;world._ext=None
            if system=='original_SARA_G0':
                with historical() as modules:
                    for record in records:
                        q=record['question_id']
                        saved[system,q]=adapters.w8(record,saved['D0',q],world.embeddings[q],ToyW8Model(tokenizer),tokenizer,
                            ToyController(),ToyScorer(),modules,device='cpu')
                continue
            for record in records:
                saved[system,record['question_id']]=dispatch(world,system,record)
    labels={r['question_id']:dict(paper_id=r['paper_id'],evaluation=dict(answers=r['gold_answers'],answers_reformatted=r['gold_answers_reformatted'])) for r in records}
    evaluated=[]
    for system in ORDER:
        rows=[saved[system,r['question_id']] for r in records]
        pair_membership(rows,[dict(question_id=r['question_id'],paper_id=r['paper_id'],split='test') for r in records])
        evaluated.extend(evaluate(rows,labels,'test'))
    from ..final_h.runner import validate_w9_rows
    groups={s:[r for r in evaluated if r['system']==s] for s in ORDER}
    validate_w9_rows(groups,dict(freeze,method=p.method_definition()))
    p.write_once(root/'sealed_predictions.json',evaluated)
    p.write_once(root/'w9h_pretest_freeze_manifest.json',dict(synthetic_only=True))
    p.write_once(root/'h2_seal.json',dict(complete=True,freeze_digest='SYNTHETIC_ONLY',artifact_hashes={}))
    with patch.object(p,'validate_freeze',return_value=freeze):result=report(root)
    accepts=[r['g2_accept'] for r in groups[p.VERIFY]]
    actions=[r['selected_action'] for r in groups['original_SARA_G0']]
    p.require(set(accepts)=={False,True} and set(actions)=={'STOP','PROMOTE'},'Synthetic branches not covered')
    return dict(synthetic_only=True,systems=ORDER,queries=2,papers=2,all_adapters_exercised=True,
        w8_branches=actions,g2_branches=accepts,fake_backend_invocations=len(toy.invocations),
        real_lm_generation_calls=0,real_encoder_calls=0,test_records_opened=False,
        fixture_membership=[r['question_id'] for r in records],aggregate_report_exercised=True,
        fixture_classification_has_no_scientific_meaning=True)
