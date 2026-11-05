"""Clean query/source-only static ranking and frozen exact-cost packing."""
import gc
import math
import time
from .protocol import *

class Scorer:
    def __init__(self,allow_download=False):
        from huggingface_hub import snapshot_download
        import torch
        from sentence_transformers import CrossEncoder
        folder=OUT/'model_cache'
        if allow_download:
            if __import__('shutil').disk_usage(HOME).free<1024**3:raise RuntimeError('Need 1 GiB disk headroom before the small model download')
            snapshot_download(MODEL,revision=REVISION,cache_dir=str(folder),
                allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','model.safetensors'])
        self.model=CrossEncoder(MODEL,revision=REVISION,cache_folder=str(folder),
            local_files_only=True,device='cpu',max_length=512,activation_fn=torch.nn.Identity(),
            model_kwargs={'dtype':torch.float32,'attn_implementation':'eager'})
        self.model.model.eval()
        for parameter in self.model.model.parameters():parameter.requires_grad_(False)
        self.info=dict(label=LABEL,model=MODEL,revision=REVISION,device=str(self.model.model.device),
            dtype=str(next(self.model.model.parameters()).dtype),parameter_count=sum(p.numel() for p in self.model.model.parameters()),
            max_length=512,activation='raw_logit_identity',training=False)
        snapshots=folder/('models--'+MODEL.replace('/','--'))/'snapshots'/REVISION
        save(OUT/'protocol/MODEL_FILES.json',dict(label=LABEL,model=MODEL,revision=REVISION,
            files={p.name:sha(p) for p in sorted(snapshots.iterdir()) if p.is_file()}))
        print('Cross-encoder:',self.info,flush=True)

    def score(self,query,sentences,passage_ids):
        """No gold, outcomes, P_help, aspects or downstream scores accepted."""
        if not isinstance(query,str) or not query.strip():raise ValueError('Nonempty query required')
        if len({s['unit_id'] for s in sentences})!=len(sentences):raise ValueError('Duplicate source sentence')
        rank={pid:i+1 for i,pid in enumerate(passage_ids)}
        pairs=[(query,s['source_text']) for s in sentences]
        start=time.perf_counter()
        import torch
        with torch.inference_mode():
            values=self.model.predict(pairs,batch_size=CONFIG['batch_size'],show_progress_bar=False,
                activation_fn=torch.nn.Identity(),apply_softmax=False,convert_to_numpy=True) if pairs else []
        if len(values)!=len(sentences):raise ValueError('Score alignment error')
        rows=[]
        for s,v in zip(sentences,values):
            v=float(v)
            if not math.isfinite(v) or s['passage_id'] not in rank:raise ValueError('Nonfinite score/unknown parent')
            pair_tokens=len(self.model.tokenizer(query,s['source_text'],truncation=False,add_special_tokens=True)['input_ids'])
            rows.append(dict(sentence_id=s['unit_id'],passage_id=s['passage_id'],passage_rank=rank[s['passage_id']],
                source_order=list(s['document_order']),source_position=s['start_char'],end_char=s['end_char'],
                sentence_text=s['source_text'],sentence_token_cost=s['token_count'],cross_encoder_score=v,
                pair_tokens_before_truncation=pair_tokens,pair_truncated=pair_tokens>512))
        return dict(label=LABEL,query_hash=digest(query),model=self.info,scores=rows,
            scoring_seconds=time.perf_counter()-start,conceptual_cross_encoder_pairs=len(pairs))

    def close(self):self.model=None;gc.collect()

def rank_scores(scores):
    return sorted(scores,key=lambda r:(-r['cross_encoder_score'],r['passage_rank'],r['source_position'],r['sentence_id']))

def pack(sentences,scores,tok,budget):
    from src.week12.algorithm import serialize
    if budget not in BUDGETS:raise ValueError('Only frozen primary budgets 256/1024')
    by={s['unit_id']:s for s in sentences}
    if len(by)!=len(sentences) or set(by)!={r['sentence_id'] for r in scores} or len(scores)!=len(sentences):
        raise ValueError('Scores must match every frozen source sentence exactly')
    selected=[];decisions=[];current=serialize([],tok)
    for i,r in enumerate(rank_scores(scores),1):
        u=by[r['sentence_id']]
        if u['source_text']!=r['sentence_text']:raise ValueError('Scored source text changed')
        trial=serialize(selected+[u],tok);dc=trial['evidence_tokens']-current['evidence_tokens']
        fits=trial['evidence_tokens']<=budget
        if fits:selected.append(u);current=trial
        decisions.append(dict(rank=i,sentence_id=u['unit_id'],score=r['cross_encoder_score'],incremental_token_cost=dc,
            selected=fits,reason='INCLUDED' if fits else 'SKIPPED_DOES_NOT_FIT',running_evidence_tokens=current['evidence_tokens']))
    # serialize sorts blocks and units back to canonical original source order.
    current=serialize(selected,tok)
    if current['evidence_tokens']>budget:raise AssertionError('Budget overflow')
    return dict(current,label=LABEL,method=METHOD,budget=budget,unused_budget=budget-current['evidence_tokens'],
        selected_unit_ids=[u['unit_id'] for u in current['units']],
        selected_passage_ids=list(dict.fromkeys(u['passage_id'] for u in current['units'])),
        selection_rank_ids=[d['sentence_id'] for d in decisions if d['selected']],decisions=decisions,
        evidence_hash=digest([current['evidence_ids'],current['text']]),stop_reason='CANDIDATES_EXHAUSTED_FIT_SKIP',
        serialization='additive-token-id-blocks-v1')
