"""Read only already-inspected DEV/FIT outputs; no fresh LM generation."""
import json
import numpy as np
import pandas as pd
from ..final_h import protocol as p
from .literal import bm25_prompt


def rows(path):
    with open(path) as f:
        for line in f:
            if line.strip():yield json.loads(line)


def upstream():
    from ..backend import load_tokenizer
    from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
    from src.week6.routing import build_routed_sara_prompt
    tokenizer=load_tokenizer(dict(checkpoint='Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42'))
    records={str(r['question_id']):r for r in rows(p.ROOT/'results/week2/raw/dev_retrieval.jsonl')}
    checks=0
    reference={}
    for r in rows(p.ROOT/'results/week7/w7b_semantic_budget/response_surface/dev_semantic_fixed_k_results.jsonl'):
        q=str(r['question_id']);rec=records[q];k=r['k']
        a=build_mixed_fidelity_sara_prompt(rec['question'].strip(),rec['passages'],r['selected_full_indices'],[],r['compressed_indices'],{},tokenizer)[0]
        b=build_routed_sara_prompt(rec['question'].strip(),rec['passages'],r['selected_full_indices'],tokenizer)[0]
        p.require(a==b,'Canonical semantic serialized prompt changed')
        reference[q,k]=r['selected_full_indices']
        checks+=1
    # Tiny identity-only inspected DEV sample, with the actual frozen encoder.
    import torch
    from src.week6.models import PretrainedSemanticScorer
    from src.week6.routing import route_r1_semantic_topk
    torch.set_num_threads(4)
    scorer=PretrainedSemanticScorer(device='cpu');fresh=0
    for q in sorted(records)[:3]:
        rec=records[q]
        scores=scorer.score_query_passages(rec['question'].strip(),[r['text'] for r in rec['passages'][:10]])
        for k in (2,4,5,6,8):
            p.require(route_r1_semantic_topk(scores,k)==reference[q,k],'Non-TEST fresh semantic allocation differs; investigate without TEST')
            fresh+=1
    del scorer
    rec=next(iter(records.values()));a=build_mixed_fidelity_sara_prompt(rec['question'].strip(),rec['passages'],list(range(8)),[],[8,9],{},tokenizer)[0]
    p.require(a==bm25_prompt()(rec['question'].strip(),rec['passages'],8,tokenizer)[0],'Canonical BM25 serialized prompt changed')
    f0={};safe={}
    for r in rows(p.F_ROOT/'fit/results.jsonl'):
        if r['policy']=='F0':f0[r['question_id']]=r
        elif r['policy']=='F2-A':safe[r['question_id']]=r
    e3={r['question_id']:r for r in rows(p.E3_ROOT/'fit/results.jsonl') if r['sentence_k']==3}
    p.require(set(e3)==set(f0)==set(safe) and len(e3)==1746,'FIT replay membership differs')
    fields=('evidence_hash','evidence_text','selected_chunk_ids','source_chunk_hashes','selected_sentence_ids')
    for q,r in e3.items():
        p.require(all(f0[q][k]==r[k] and safe[q][k]==r[k] for k in fields),'Saved E3/F evidence parity differs')
        p.require(f0[q]['prediction']==r['prediction'],'Saved F0 alias answer differs')
    from ..action_verification_g.features import payload,online_features
    feature_schema=p.read(p.G_ROOT/'final_fit/feature_schema.json')['feature_columns']
    saved=pd.read_csv(p.G_ROOT/'scaffold/inference_features.csv',dtype={'question_id':str,'paper_id':str}).set_index('question_id')
    for q in f0:
        d0=dict(prediction=f0[q]['initial_draft'])
        current=online_features(payload(d0,f0[q],e3[q]))
        p.require(np.allclose([current[k] for k in feature_schema],saved.loc[q,feature_schema].to_numpy(float),rtol=0,atol=1e-12,equal_nan=True),'G2 feature replay differs')
    p.require(p.file_hash(p.G_ROOT/'final_fit/logistic_pipeline.joblib')==p.file_hash(p.G_ROOT/'controller_fit_oof/logistic.joblib'),'G2 deployment model differs from audited full-FIT model')
    return dict(test_accessed=False,actual_generation_calls=0,semantic_dev_prompt_matches=checks,
        fresh_semantic_dev_queries=3,fresh_semantic_budget_allocation_matches=fresh,bm25_dev_prompt_match=True,
        f0_fit_answer_alias_matches=1746,f2_fit_evidence_identity_matches=1746,g2_fit_feature_replay_matches=1746,
        g2_serialized_pipeline_equivalent=True,
        limitation='Saved answers/identity/features and literal prompt parity; no fresh GPU numerical regeneration and no new all-FIT performance estimate.')
