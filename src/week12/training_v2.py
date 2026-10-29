"""Uncapped frozen TRAIN corpus, one evidence condition/query, VAL-only selection."""
import json,math
from collections import Counter
from .protocol import ROOT,root,identity,immutable,atomic,digest,sha
from .splits import read_freeze,load_examples

def condition(cfg,dataset,query_id,role='TRAIN'):
    modes=[(m,b) for m in cfg['lora']['modes'] for b in cfg['budgets']]
    return modes[int(digest([cfg['seed'],dataset,str(query_id)]),16)%len(modes)] if role=='TRAIN' else ('QBGF_FULL',256)

def balanced_smoke_sampler(rows,accumulation=16,seed=42):
    import torch
    counts=Counter(r['dataset'] for r in rows)
    if not counts:raise ValueError('Empty TRAIN corpus')
    weights=[1/counts[r['dataset']] for r in rows]
    indices=list(torch.utils.data.WeightedRandomSampler(weights,len(rows),replacement=True,generator=torch.Generator().manual_seed(seed)))
    actual=Counter(rows[i]['dataset'] for i in indices);touched={d:0 for d in counts}
    for start in range(0,len(indices),accumulation):
        for d in {rows[i]['dataset'] for i in indices[start:start+accumulation]}:touched[d]+=1
    return indices,dict(raw_eligible_examples=dict(counts),sampling_probabilities={d:1/len(counts) for d in counts},effective_samples_per_epoch=len(indices),planned_sample_counts=dict(actual),planned_optimizer_updates_with_dataset=touched,planned_fractional_update_contribution={d:v/accumulation for d,v in actual.items()},note='All eligible queries are in the sampling pool. Replacement balancing need not visit every unique query in one epoch. Planned counts are not claimed as completed updates.')

def balancing(rows,accumulation=16,seed=42):
    from .scheduling import full_coverage
    return full_coverage(rows,accumulation,seed)

def prepare_corpus(cfg):
    from transformers import AutoTokenizer
    from .laboratory import Lab
    from .algorithm import choose,validate_spans
    from .reader import prompt_ids
    from .protocol import Store
    frozen=read_freeze(cfg);out=root(cfg)/'lora/corpus';store=Store(out/'shards',identity(cfg,dict(split_hash=frozen['split_hash'],sampling=cfg['lora']['evidence_mode_sampling'])))
    tok=AutoTokenizer.from_pretrained(cfg['models']['Qwen']['id'],revision=cfg['models']['Qwen']['revision'],local_files_only=True);lab=Lab(cfg,tok)
    manifest=[]
    try:
        for role in ('TRAIN','VAL'):
            for d in cfg['datasets']:
                for e in load_examples(cfg,role,d):
                    mode,budget=condition(cfg,d,e['query_id'],role);key=[d,e['query_id'],role];manifest.append(dict(dataset=d,query_id=e['query_id'],cluster_id=e['cluster_id'],partition='train' if role=='TRAIN' else 'validation',mode=mode,budget=budget))
                    if store.get(key) is not None:continue
                    pool,sent,props,ps,ss,_=lab.prepare(e);sel=choose(pool,sent,props,ps,ss,tok,budget,mode);validate_spans(sel['units'],e['inference']['documents']);prompt=prompt_ids(tok,e['question'],e['task_type'],sel['evidence_ids'],cfg['labels'])
                    target=e['evaluation']['gold_label'] if d=='SciFact' else str(e['evaluation']['gold_answer'][0]);target_ids=tok.encode(target,add_special_tokens=False)+[tok.eos_token_id]
                    if len(prompt)+len(target_ids)>cfg['lora']['max_length']:raise ValueError('Sequence exceeds preregistered cap; no silent drop or truncation. Review before training in a new protocol version.')
                    store.put(key,dict(manifest[-1],input_ids=prompt+target_ids,labels=[-100]*len(prompt)+target_ids,attention_mask=[1]*(len(prompt)+len(target_ids)),prompt_ids=prompt,target=target,references=e['evaluation']['gold_answer'],task_type=e['task_type'],evidence_hash=sel['evidence_hash'],partition_role=role))
                    lab.units_cache.clear();lab.ranks.clear()
    finally:lab.close()
    rows=store.rows();validate_rows(cfg,rows);immutable(out/'membership.json',manifest);immutable(out/'rows.json',rows)
    indices,plan=balancing([r for r in rows if r['partition']=='train'],cfg['lora']['gradient_accumulation_steps'],cfg['seed']);plan.update(groups={d:len({r['cluster_id'] for r in rows if r['dataset']==d and r['partition']=='train'}) for d in cfg['datasets']},mode_counts=dict(Counter(str((r['dataset'],r['mode'],r['budget'])) for r in rows if r['partition']=='train')),estimated_optimizer_steps=math.ceil(len(indices)/cfg['lora']['gradient_accumulation_steps']),max_tokens=max(len(r['input_ids']) for r in rows),min_tokens=min(len(r['input_ids']) for r in rows),mean_tokens=sum(len(r['input_ids']) for r in rows)/len(rows));atomic(out/'training_plan.json',plan);return rows

def validate_rows(cfg,rows):
    f=read_freeze(cfg)
    for d,p in f['datasets'].items():
        for role,part in [('TRAIN','train'),('VAL','validation')]:
            seen=[r for r in rows if r['dataset']==d and r['partition']==part];wanted={r['query_id']:r for r in p[role]}
            if len(seen)!=len(wanted) or {r['query_id'] for r in seen}!=set(wanted):raise ValueError('Corpus must contain exactly ALL frozen '+d+' '+role+' queries, once each')
            for r in seen:
                if r.get('partition_role')!=role or r['cluster_id']!=wanted[r['query_id']]['group']:raise ValueError('Gradient/validation role or group changed')
    if any(r['partition'] not in ('train','validation') for r in rows):raise ValueError('Confirmation cannot enter corpus')
    return True

def validation_metrics(model,tok,rows,cfg):
    """All frozen VAL only. Balanced answer NLL chooses checkpoint; task scores diagnose."""
    import torch,numpy as np
    from .reader import parse_label
    from .statistics import metric
    from src.week6.evaluation import compute_qa_answer_metrics
    from src.week11.evaluation import qa_pair
    validate_roles={r.get('partition_role') for r in rows}
    if validate_roles!={'VAL'}:raise ValueError('Checkpoint selection requires VAL only')
    per={d:[] for d in cfg['datasets']};was_training=model.training;model.eval()
    with torch.inference_mode():
        for r in rows:
            x=torch.tensor([r['input_ids']],device=model.device);labels=torch.tensor([r['labels']],device=model.device)
            loss=float(model(input_ids=x,attention_mask=torch.ones_like(x),labels=labels).loss)
            prompt=torch.tensor([r['prompt_ids']],device=model.device);cap=cfg['generation']['max_new_tokens_verification' if r['dataset']=='SciFact' else 'max_new_tokens_qa']
            pred=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),max_new_tokens=cap,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id);text=tok.decode(pred[0,len(r['prompt_ids']):],skip_special_tokens=True).strip()
            if r['dataset']=='SciFact':p=parse_label(text,cfg['labels']);obs=dict(nll=loss,prediction=p,gold=r['target'],correct=float(p==r['target']),format_compliant=p!='INVALID_OUTPUT')
            else:
                f1=compute_qa_answer_metrics(text,r['references'])['token_f1'] if r['dataset']=='QASPER' else max(qa_pair(text,a)[0] for a in r['references'])
                obs=dict(nll=loss,f1=f1,prediction=text,format_compliant=True)
            per[r['dataset']].append(obs)
    panel={d:dict(n=len(v),answer_nll=float(np.mean([r['nll'] for r in v])),task_score=metric(v,cfg['statistics']['primary_metric'][d]),format_compliance=float(np.mean([r['format_compliant'] for r in v]))) for d,v in per.items() if v}
    model.train(was_training)
    return dict(eval_loss=float(np.mean([v['answer_nll'] for v in panel.values()])),validation_task_scores=panel,selection_objective='equal-dataset mean query-mean answer NLL; minimize; ties earliest',data_role='VAL')
