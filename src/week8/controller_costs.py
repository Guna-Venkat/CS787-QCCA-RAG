"""Exact tokenizer input lengths for cached single-promotion replay (CPU only)."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from src.week8.data import PROJECT_ROOT,load_config
from src.week8.generator_features import source_fingerprint,sha256_file
from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
from src.week6.routing import build_routed_sara_prompt

KEYS=['question_id','paper_id','semantic_rank','candidate_passage_index']


def apply_exact_token_costs(surface,split):
    folder=PROJECT_ROOT/'results/week8/w8e_prompt_tokens';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{split}.csv';meta=folder/f'{split}_provenance.json'
    data_fingerprint,_=source_fingerprint(split)
    raw=PROJECT_ROOT/'results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl'
    fingerprint=hashlib.sha256((data_fingerprint+Path(__file__).read_text()+sha256_file(raw)+
        sha256_file(PROJECT_ROOT/'src/week6/routing.py')).encode()).hexdigest()
    if path.exists():
        provenance=json.loads(meta.read_text())
        if provenance['source_fingerprint']!=fingerprint or provenance['csv_sha256']!=sha256_file(path):
            raise ValueError('Exact prompt-token cache provenance changed')
        costs=pd.read_csv(path,dtype={'question_id':str,'paper_id':str})
    else:
        cfg=load_config();tokenizer=AutoTokenizer.from_pretrained(str(PROJECT_ROOT/cfg['sara']['checkpoint_path']),use_fast=True)
        wanted=set(surface.question_id);records={};allocations={}
        with (PROJECT_ROOT/cfg['splits']['train_retrieval']).open() as f:
            for line in f:
                r=json.loads(line)
                if str(r['question_id']) in wanted:records[str(r['question_id'])]=r
        with raw.open() as f:
            for line in f:
                r=json.loads(line)
                if str(r['question_id']) in wanted and int(r['k']) in (4,6):allocations[(str(r['question_id']),int(r['k']))]=r['selected_full_indices']
        rows=[]
        def count(prompt):return len(tokenizer(prompt)['input_ids'])
        for q,g in surface.groupby('question_id'):
            record=records[q];passages=record['passages'][:10]
            question=str(record['question']).strip()
            full=sorted(set(range(10))-set(g.candidate_passage_index.astype(int)))
            prompt,_,_,_=build_mixed_fidelity_sara_prompt(question,passages,full,[],sorted(set(range(10))-set(full)),{},tokenizer,hybrid_medium=False)
            original,_,_=build_routed_sara_prompt(question,passages,allocations[(q,4)],tokenizer)
            if tokenizer(prompt)['input_ids']!=tokenizer(original)['input_ids']:
                raise ValueError(f'{q}: rebuilt k4 prompt differs from canonical W7 prompt')
            k6,_,_=build_routed_sara_prompt(question,passages,allocations[(q,6)],tokenizer)
            base_tokens=count(prompt);k6_tokens=count(k6)
            for row in g.itertuples():
                f=sorted(full+[int(row.candidate_passage_index)]);heavy=sorted(set(range(10))-set(f))
                promoted,_,_,_=build_mixed_fidelity_sara_prompt(question,passages,f,[],heavy,{},tokenizer,hybrid_medium=False)
                rows.append({**{k:getattr(row,k) for k in KEYS},'exact_base_prompt_tokens':base_tokens,
                    'exact_promoted_prompt_tokens':count(promoted),'exact_k6_prompt_tokens':k6_tokens})
        costs=pd.DataFrame(rows);costs.to_csv(path,index=False)
        meta.write_text(json.dumps({'source_fingerprint':fingerprint,'csv_sha256':sha256_file(path),
            'rows':len(costs),'includes_chat_template_and_special_tokens':True},indent=2))
    result=surface.merge(costs,on=KEYS,how='left',validate='one_to_one',sort=False)
    columns=['exact_base_prompt_tokens','exact_promoted_prompt_tokens','exact_k6_prompt_tokens']
    if len(result)!=len(surface) or not np.isfinite(result[columns].to_numpy()).all():raise ValueError('Incomplete exact token costs')
    result['historical_base_context_tokens']=result.base_context_tokens
    result['historical_promoted_context_tokens']=result.promoted_context_tokens
    # Resource accounting must not redefine numeric features used by the
    # already-trained utility model. Exact costs remain in separate columns.
    return result
