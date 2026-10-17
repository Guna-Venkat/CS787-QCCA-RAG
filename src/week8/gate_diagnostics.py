"""Measured canonical-draft token diagnostics for the FIT-only B0-GW gate."""
from __future__ import annotations
import hashlib,json,time,inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from src.week8.data import PROJECT_ROOT,load_config,load_week8_split
from src.week8.utility_training import load_surface
from src.week8.generator_features import source_fingerprint,canonical_k4_context
from src.week8.semantic import SemanticScorer
from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt,extract_mixed_fidelity_sara_embeddings

U_COLUMNS=['u1_entropy_mean','u2_entropy_max','u3_margin_mean','u4_margin_min',
 'u5_logprob_mean','u6_seq_logprob','u7_answer_len','u8_eos_conf','u9_empty_answer',
 'u10_repetition_ratio','u11_lexical_overlap','u12_semantic_sim']


def logits_diagnostics(logits,token_ids,eos_id):
    logits=logits.float();ids=torch.as_tensor(token_ids,device=logits.device)
    if logits.ndim!=2 or len(logits)!=len(ids) or not len(ids):raise ValueError('One logits row per actual token is required')
    lp=torch.log_softmax(logits,dim=-1);prob=lp.exp()
    entropy=-(prob*lp).sum(-1);top=torch.topk(logits,2,dim=-1).values
    selected=lp.gather(1,ids[:,None]).squeeze(1)
    return dict(token_ids=ids.cpu().tolist(),token_entropies=entropy.cpu().tolist(),
        top1_top2_margins=(top[:,0]-top[:,1]).cpu().tolist(),token_logprobs=selected.cpu().tolist(),
        eos_confidence=float(prob[-1,eos_id]),greedy_verified=bool(torch.equal(logits.argmax(-1),ids)))


def summarize_diagnostics(raw,question,draft,semantic_similarity):
    lp=raw['token_logprobs'];ent=raw['token_entropies'];margin=raw['top1_top2_margins'];ids=raw['token_ids']
    if not (len(ids)==len(ent)==len(margin)==len(lp)>0):raise ValueError('Incomplete token uncertainty measurements')
    q,d=set(question.lower().split()),set(draft.lower().split())
    result=dict(zip(U_COLUMNS,[np.mean(ent),np.max(ent),np.mean(margin),np.min(margin),
        np.mean(lp),np.sum(lp),len(ids),raw['eos_confidence'],int(not draft.strip()),
        1-len(set(ids))/len(ids),len(q&d)/max(len(q|d),1),semantic_similarity]))
    result={k:float(v) for k,v in result.items()}
    if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite gate features')
    return result


def measure_canonical_draft(model,tokenizer,prompt,draft,embeds,device='cuda'):
    """Recover a greedy-consistent token stream; never silently replace the draft.

    Whole prompt+continuation tokenization avoids the incorrect first token
    produced by separately tokenizing a stripped answer. Prefix equality and
    every greedy token are checked. Ambiguous reconstruction falls back to actual
    greedy generation, whose decoded answer must match the frozen canonical text.
    """
    prefix=tokenizer(prompt,return_tensors='pt')['input_ids']
    embeds=embeds.to(device=device,dtype=model.dtype)
    started=time.perf_counter()
    for space in ['', ' ']:
        if not draft and space:continue
        full=tokenizer(prompt+space+draft,return_tensors='pt')['input_ids']
        if full.shape[1]<prefix.shape[1] or not torch.equal(full[:,:prefix.shape[1]],prefix):continue
        targets=full[0,prefix.shape[1]:].tolist()
        if len(targets)<128:targets.append(tokenizer.eos_token_id)
        if not targets or len(targets)>128:continue
        ids=torch.cat([prefix,torch.tensor([targets])],1).to(device)
        # Only the answer-predicting positions need vocabulary logits. This is a
        # single parallel forward, not regeneration of W8-A promotions.
        with torch.inference_mode():
            output=model(input_ids=ids,attention_mask=torch.ones_like(ids),retrieval_embeds=embeds,
                         use_cache=False,logits_to_keep=len(targets)+1)
            stats=logits_diagnostics(output.logits[0,:len(targets)],targets,tokenizer.eos_token_id)
            del output
        if stats['greedy_verified']:
            return {**stats,'diagnostic_source':'teacher_forced_greedy_verified',
                    'diagnostic_seconds':time.perf_counter()-started}
    with torch.inference_mode():
        ids=prefix.to(device)
        output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),retrieval_embeds=embeds,
            max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,return_dict_in_generate=True,output_scores=True)
        prediction=tokenizer.decode(output.sequences[0],skip_special_tokens=True).strip()
        if prediction!=draft.strip():
            raise ValueError(f'Regenerated draft differs from canonical: {prediction[:100]!r} vs {draft[:100]!r}; do not replace frozen outcomes')
        if len(output.scores)!=output.sequences.shape[1]:raise ValueError('SARA continuation/score alignment changed')
        logits=torch.stack([score[0] for score in output.scores])
        stats=logits_diagnostics(logits,output.sequences[0],tokenizer.eos_token_id)
        del output
    if not stats['greedy_verified']:raise ValueError('Selected token/logit alignment is invalid')
    return {**stats,'diagnostic_source':'regenerated_canonical_match',
            'diagnostic_seconds':time.perf_counter()-started}


def measure_draft_batch(model,tokenizer,items,device='cuda'):
    """Actual greedy generation scores, checked against each frozen draft.

    Batching shares GPU weight reads; actual EOS terminates each token record.
    Padding scores after EOS are never treated as answer-token measurements.
    A batch-numerical mismatch is retried individually, never silently accepted.
    """
    prefixes=[tokenizer(item['prompt'],return_tensors='pt')['input_ids'][0] for item in items]
    width=max(len(ids) for ids in prefixes)
    ids=torch.full((len(items),width),tokenizer.pad_token_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids)
    for i,prefix in enumerate(prefixes):
        ids[i,-len(prefix):]=prefix.to(device);mask[i,-len(prefix):]=1
    embeds=torch.cat([item['embeds'] for item in items],0).to(device=device,dtype=model.dtype)
    started=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(input_ids=ids,attention_mask=mask,retrieval_embeds=embeds,
            max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,return_dict_in_generate=True,output_scores=True)
        if len(output.scores)!=output.sequences.shape[1]:raise ValueError('SARA continuation/score alignment changed')
        result={};mismatched=[]
        for i,item in enumerate(items):
            tokens=output.sequences[i].tolist()
            if tokenizer.eos_token_id in tokens:tokens=tokens[:tokens.index(tokenizer.eos_token_id)+1]
            decoded=tokenizer.decode(tokens,skip_special_tokens=True).strip()
            if decoded!=item['draft_answer'].strip():
                mismatched.append(item);continue
            logits=torch.stack([score[i] for score in output.scores[:len(tokens)]])
            stats=logits_diagnostics(logits,tokens,tokenizer.eos_token_id)
            if not stats['greedy_verified']:raise ValueError('Greedy token/logit alignment mismatch')
            result[item['question_id']]={**stats,'diagnostic_source':'greedy_canonical_match'}
        del output
    duration=(time.perf_counter()-started)/len(items)
    for q in result:result[q]['diagnostic_seconds']=duration
    for item in mismatched:
        if len(items)==1:
            raise ValueError(f"Regenerated draft differs from canonical for query {item['question_id']}; stop rather than change frozen outcomes")
        retry=measure_draft_batch(model,tokenizer,[item],device)
        retry[item['question_id']]['diagnostic_seconds']+=duration
        result.update(retry)
    return result


def gate_source_fingerprint(split):
    """Hash measured inputs and measurement definitions, independent of scheduling."""
    base,_=source_fingerprint(split)
    algorithm=hashlib.sha256(''.join(inspect.getsource(f) for f in
        [logits_diagnostics,summarize_diagnostics,measure_draft_batch]).encode()).hexdigest()
    current=hashlib.sha256((base+algorithm).encode()).hexdigest()
    legacy=[]
    # Explicit migration: these existing rows used identical measurement
    # functions and data; only processing order now uses exact draft lengths.
    if split=='fit' and base=='9a7cb67d51d6767757cd85d2826556f47942d56b898717d056adc056d517fa16' and algorithm=='dbe0567e4e89d43ecc79aae30195e68af4df9021921a0e68d5300cb63b43d292':
        legacy=['3b1185361e9e0b5c311460cbc3a0fa98640dced934a33754b1dd2def92df83fe']
    return current,base,algorithm,legacy


def backfill_gate_inputs(split,output_dir,device='cuda',limit=None):
    surface=load_surface(split);groups=list(surface.groupby('question_id',sort=True))
    if limit is not None:groups=groups[:limit]
    source,existing_fingerprint,measurement_fingerprint,legacy_sources=gate_source_fingerprint(split)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/f'canonical_diagnostics_{split}.jsonl';done={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line);qid=str(row['question_id'])
            if qid in done:raise ValueError('Duplicate canonical diagnostics')
            if row['source_fingerprint'] not in [source,*legacy_sources]:raise ValueError('Gate input source changed; use a fresh cache')
            done[qid]=row
    refs={str(q):g for q,g in groups}
    if set(done)-set(refs):raise ValueError('Gate cache has queries outside this run')
    for q,r in done.items():
        if r['draft_answer']!=str(refs[q].iloc[0].draft_answer) or r['paper_id']!=str(refs[q].iloc[0].paper_id):raise ValueError('Cached gate draft/identity differs')
        summarize_diagnostics(r,r['question'],r['draft_answer'],r['features']['u12_semantic_sim'])
    pending=[(str(q),g) for q,g in groups if str(q) not in done]
    lengths=pd.read_csv(PROJECT_ROOT/f'results/week8/w8d_attribution/full/features_{split}.csv',dtype={'question_id':str}).groupby('question_id').masking_target_tokens.first().to_dict()
    pending.sort(key=lambda item:(int(lengths[item[0]])//4,int(item[1].iloc[0].base_context_tokens)//256))
    if pending:
        records,sfr,_=load_week8_split(split);records={str(r['question_id']):r for r in records}
        if device.startswith('cuda') and torch.cuda.mem_get_info(device)[0]<18*1024**3:raise RuntimeError('Need 18 GiB free GPU memory for SARA diagnostic backfill')
        import src
        baseline=str(PROJECT_ROOT/'Baselines/SARA-main/src')
        if baseline not in src.__path__:src.__path__.append(baseline)
        from src.model.loader import load_sara_for_eval
        cfg=load_config();model,tokenizer=load_sara_for_eval(str(PROJECT_ROOT/cfg['sara']['checkpoint_path']),device=device)
        scorer=SemanticScorer(device='cpu')
        try:
            with path.open('a') as f:
                start=0
                progress=tqdm(total=len(pending),desc=f'Canonical token diagnostics {split}',unit='query')
                while start<len(pending):
                    size=min(8,len(pending)-start)
                    while size>1 and max(int(g.iloc[0].base_context_tokens)+2*len(str(g.iloc[0].draft_answer).split())+256 for _,g in pending[start:start+size])*size>8192:
                        size=max(1,size//2)
                    batch=[]
                    for q,g in pending[start:start+size]:
                        record=records[q]
                        question,prompt,_,compressed,embeds,_=canonical_k4_context(g,record,sfr[q],tokenizer)
                        batch.append({'question_id':q,'group':g,'question':question,
                            'draft_answer':str(g.iloc[0].draft_answer),'prompt':prompt,
                            'embeds':embeds})
                    measured=measure_draft_batch(model,tokenizer,batch,device)
                    for item in batch:
                        q=item['question_id'];g=item['group'];draft=item['draft_answer'];question=item['question']
                        stats=measured[q]
                        semantic=float(scorer.score_query_passages(question,[draft]).cpu()[0]) if draft else 0.
                        features=summarize_diagnostics(stats,question,draft,semantic)
                        row={'question_id':q,'paper_id':str(g.iloc[0].paper_id),'question':question,
                            'draft_answer':draft,'source_fingerprint':source,'features':features,**stats}
                        f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();done[q]=row
                    start+=size;progress.update(size)
                progress.close()
        finally:
            del model,tokenizer
            if torch.cuda.is_available():torch.cuda.empty_cache()
    result=pd.DataFrame([{'question_id':str(q),'paper_id':str(g.iloc[0].paper_id),**done[str(q)]['features']} for q,g in groups])
    result.to_csv(output_dir/f'gate_features_{split}.csv',index=False)
    (output_dir/f'provenance_{split}.json').write_text(json.dumps({'source_fingerprint':source,
        'input_w8d_fingerprint':existing_fingerprint,'queries':len(groups),'feature_columns':U_COLUMNS,
        'features_are_measured':True,'split':split,'gold_in_features':False,
        'diagnostic_backend':'actual greedy continuation tokens and full-vocabulary scores; canonical decoded answer checked',
        'measurement_fingerprint':measurement_fingerprint,'compatible_prior_fingerprints':legacy_sources,
        'compatibility_note':'Retained prior actual-greedy rows have identical measurement functions and input hashes; scheduling-only changes do not invalidate measurements.'},indent=2))
    return result
