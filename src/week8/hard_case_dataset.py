"""FIT-only hard-case supervision on canonical semantic passage identities.

Outcome labels may choose TRAINING contexts and weights. Validation contexts
must be constructed separately from frozen predictions, never these categories.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
from src.week8.qcla_training import gold_target_inputs

CASE_TYPES=['TYPE_A_HELPFUL','TYPE_B_DISTRACTOR_SAFE','TYPE_C_NEUTRAL']


def categorize_hard_cases(promotion_records_by_qid,epsilon_help=.01,epsilon_hurt=-.01):
    result={}
    for qid,rows in promotion_records_by_qid.items():
        if not rows:raise ValueError('Missing promotion candidates')
        if any('delta_f1' not in r or 'candidate_passage_index' not in r for r in rows):
            raise ValueError('Real deltas and original passage identities are required')
        values=np.asarray([r['delta_f1'] for r in rows],dtype=float)
        if not np.isfinite(values).all():raise ValueError('Nonfinite promotion outcomes')
        # Stable ties: semantic rank, then original passage index.
        ordered=sorted(rows,key=lambda r:(int(r.get('semantic_rank',11)),int(r['candidate_passage_index'])))
        best=max(ordered,key=lambda r:float(r['delta_f1']))
        maximum=float(values.max());minimum=float(values.min())
        kind=CASE_TYPES[0] if maximum>epsilon_help else (CASE_TYPES[1] if minimum<epsilon_hurt else CASE_TYPES[2])
        result[str(qid)]={'question_id':str(qid),'case_type':kind,'max_delta':maximum,'min_delta':minimum,
            'best_candidate_idx':int(best['candidate_passage_index']),'num_candidates':len(rows)}
    return result


def compute_outcome_weights(delta,gamma=2.):
    if not np.isfinite([delta,gamma]).all() or gamma<0:raise ValueError('Invalid outcome weight')
    return 1.+gamma*max(0.,float(delta))


def build_hard_case_dataset(raw_queries,promotion_records,tokenizer=None,mode='hard_case',gamma=2.,
        type_a_weight=2.,type_b_weight=2.,type_c_weight=1.,epsilon_help=.01,epsilon_hurt=-.01):
    if tokenizer is None:raise ValueError('The real SARA tokenizer is required')
    if mode not in ['uniform','hard_case','outcome_weighted']:raise ValueError('Unknown supervision mode')
    query_map={str(r['question_id']):r for r in raw_queries}
    if len(query_map)!=len(raw_queries):raise ValueError('Duplicate training query')
    frame=pd.DataFrame(promotion_records)
    if frame.empty:raise ValueError('Missing FIT promotion surface; no fabricated examples')
    frame['question_id']=frame.question_id.astype(str)
    groups={q:g for q,g in frame.groupby('question_id',sort=True)}
    if set(groups)!=set(query_map):raise ValueError('Promotion coverage must exactly match training query IDs')
    categories=categorize_hard_cases({q:g.to_dict('records') for q,g in groups.items()},epsilon_help,epsilon_hurt)
    examples=[];counts=Counter({k:0 for k in CASE_TYPES})
    for q in sorted(query_map):
        r=query_map[q];g=groups[q];c=categories[q]
        if len(g)!=6 or set(g.semantic_rank.astype(int))!=set(range(5,11)) or g.candidate_passage_index.nunique()!=6:
            raise ValueError(f'{q}: expected all six canonical semantic candidates')
        if not g.candidate_passage_index.between(0,9).all():raise ValueError('Invalid original passage index')
        if not g.paper_id.astype(str).eq(str(r['paper_id'])).all():raise ValueError('Training paper identity mismatch')
        passages=r['passages'][:10]
        if len(passages)!=10:raise ValueError('Exactly ten retrieved passages required')
        for row in g.itertuples():
            text=passages[int(row.candidate_passage_index)]['text']
            if str(row.candidate_text).strip()!=str(text).strip():raise ValueError('Candidate text differs from original passage')
        baseline_heavy=sorted(g.candidate_passage_index.astype(int).tolist())
        baseline_full=sorted(set(range(10))-set(baseline_heavy))
        selected=c['best_candidate_idx'] if c['case_type']==CASE_TYPES[0] else None
        full=sorted(baseline_full+([selected] if selected is not None else []))
        heavy=sorted(set(range(10))-set(full))
        prompt,_,compressed,_=build_mixed_fidelity_sara_prompt(str(r['question']).strip(),passages,full,[],heavy,{},tokenizer,hybrid_medium=False)
        if compressed!=heavy:raise ValueError('Compressed-slot order changed')
        kind=c['case_type'];counts[kind]+=1
        if mode=='uniform':weight=1.
        elif mode=='hard_case':weight={CASE_TYPES[0]:type_a_weight,CASE_TYPES[1]:type_b_weight,CASE_TYPES[2]:type_c_weight}[kind]
        else:weight=compute_outcome_weights(c['max_delta'] if kind==CASE_TYPES[0] else (-c['min_delta'] if kind==CASE_TYPES[1] else 0.),gamma)
        if not np.isfinite(weight) or weight<=0:raise ValueError('Invalid measured training weight')
        examples.append({'question_id':q,'paper_id':str(r['paper_id']),'case_type':kind,'record':r,
            'prompt_text':prompt,'gold_text':r['gold_answers'][0],'weight':float(weight),'raw_weight':float(weight),
            'baseline_full_indices':baseline_full,'full_indices':full,'compressed_indices':compressed,
            'training_promoted_passage_index':selected,'max_delta':c['max_delta'],'min_delta':c['min_delta']})
    return examples,dict(counts)


class HardCasePromptDataset(Dataset):
    def __init__(self,examples,tokenizer,max_seq_len=4096):
        if not examples:raise ValueError('Empty real training dataset')
        self.examples=examples;self.tokenizer=tokenizer;self.max_seq_len=max_seq_len
    def __len__(self):return len(self.examples)
    def __getitem__(self,idx):
        ex=self.examples[idx]
        ids,targets=gold_target_inputs(self.tokenizer,ex['prompt_text'],ex['record'],'cpu')
        if ids.shape[1]>self.max_seq_len:raise ValueError('Input exceeds training limit; never truncate away targets or compression slots')
        return {**ex,'input_ids':ids,'targets':targets,'attention_mask':torch.ones_like(ids),
            'weight':float(ex['weight'])}
