"""Manual cache/smoke/FIT-screen only. No confirmation data or SARA latent calls."""
from __future__ import annotations
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import yaml
from .provenance import ROOT,branch_guard,digest,file_hash
from .caching import atomic_json,RowCache,read_jsonl
from .model_registry import MODELS,CANONICAL,get_model
from .model_runtime import TextRuntime
from .config import load_config
from .data import load_records,resolve
from .evaluation import answer_f1
from .faithfulness import audit_extraction,supported_evidence


def verify_snapshot(path):
    path=Path(path)
    for name in ['config.json','tokenizer_config.json']:
        if not (path/name).is_file():raise FileNotFoundError('Incomplete snapshot: '+name)
    if not any((path/x).is_file() for x in ['tokenizer.json','tokenizer.model']):raise FileNotFoundError('Missing tokenizer vocabulary')
    index=path/'model.safetensors.index.json'
    weights=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    for name in weights:
        p=path/name
        if not p.is_file() or p.stat().st_size<=0:raise FileNotFoundError('Missing model shard: '+name)
    return sum((path/n).stat().st_size for n in weights)


def find_cached_snapshot(model_id,cache_dir=None):
    from huggingface_hub.constants import HF_HUB_CACHE
    folder=Path(cache_dir or HF_HUB_CACHE)/('models--'+model_id.replace('/','--'))
    ref=folder/'refs/main'
    paths=[]
    if ref.exists():paths.append(folder/'snapshots'/ref.read_text().strip())
    paths.extend(sorted((folder/'snapshots').glob('*')))
    for path in dict.fromkeys(paths):
        try:return path,verify_snapshot(path)
        except FileNotFoundError:
            continue  # An incomplete local snapshot is never declared cached.
    raise FileNotFoundError('No complete official snapshot in cache: '+model_id)


def cache_models(cfg,models,local_only=False):
    from huggingface_hub import HfApi,snapshot_download,hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError,LocalEntryNotFoundError,OfflineModeIsEnabled
    import requests
    output=resolve(cfg['output_root']);output.mkdir(parents=True,exist_ok=True)
    rows=[];cache_dir=cfg.get('cache_dir')
    for value in models:
        spec=get_model(value);row=dict(model_id=spec.model_id,revision=None,local_cache_path=None,
            download_status='UNAVAILABLE',disk_size=None,license=spec.license,gated=spec.gated,access_status='unverified')
        try:
            try:
                path,size=find_cached_snapshot(spec.model_id,cache_dir)
            except (FileNotFoundError,LocalEntryNotFoundError):
                if local_only:
                    row.update(download_status='NOT_CACHED',access_status='not checked (local-only)');rows.append(row);continue
                api=HfApi();info=api.model_info(spec.model_id,files_metadata=True,timeout=15)
                revision=info.sha
                # Fetch the tiny index before downloading only the necessary weight shards.
                names={s.rfilename:s.size for s in info.siblings}
                if 'model.safetensors.index.json' in names:
                    index=hf_hub_download(spec.model_id,'model.safetensors.index.json',revision=revision,cache_dir=cache_dir)
                    weights=sorted(set(json.loads(Path(index).read_text())['weight_map'].values()))
                else:weights=['model.safetensors']
                if any(n not in names or names[n] is None for n in weights):raise ValueError('Official checkpoint size/weights not verifiable')
                needed=sum(names[n] for n in weights)
                from huggingface_hub.constants import HF_HUB_CACHE
                disk=Path(cache_dir or HF_HUB_CACHE);disk.mkdir(parents=True,exist_ok=True)
                free=shutil.disk_usage(disk).free
                row.update(revision=revision,official_weight_bytes=needed,free_disk_before=free)
                if needed+cfg['min_free_disk_bytes']>free:
                    row.update(download_status='INSUFFICIENT_DISK',access_status='metadata accessible');rows.append(row);continue
                extras=[n for n in names if '/' not in n and (n.endswith(('.json','.model','.jinja','.txt')) or n.startswith(('LICENSE','README','tokenizer')))]
                path=Path(snapshot_download(spec.model_id,revision=revision,cache_dir=cache_dir,allow_patterns=weights+extras,max_workers=2))
                size=verify_snapshot(path)
            row.update(revision=path.name,local_cache_path=str(path),download_status='COMPLETE',disk_size=size,
                access_status='existing official snapshot verified' if local_only else 'official snapshot verified')
        except (HfHubHTTPError,OfflineModeIsEnabled,requests.RequestException,OSError,ValueError) as exc:
            row.update(download_status='ACCESS_OR_NETWORK_ERROR',access_status=type(exc).__name__,error=str(exc)[:400])
        rows.append(row)
    manifest_path=output/'model_cache_manifest.json'
    previous=json.loads(manifest_path.read_text())['models'] if manifest_path.exists() else []
    merged={r['model_id']:r for r in previous}
    merged.update({r['model_id']:r for r in rows})
    branch_guard();atomic_json(manifest_path,{'models':list(merged.values()),'one_model_at_a_time':True,'no_cache_deleted':True})
    return rows


def freeze_subset(cfg,records):
    """Whole FIT papers, deterministic type/difficulty interleaving; no model results."""
    path=resolve(cfg['output_root'])/'preflight_query_ids.json'
    identities={r['inference']['question_id']:r['inference']['paper_id'] for r in records}
    source=digest(identities)
    if path.exists():
        value=json.loads(path.read_text())
        if value['fit_membership_hash']!=source or value['seed']!=cfg['seed']:raise ValueError('Frozen FIT subset identity changed')
        if any(q not in identities for q in value['question_ids']):raise ValueError('Subset contains non-FIT queries')
        return value
    bypaper={};strata={}
    for r in records:
        inf=r['inference'];pid=inf['paper_id'];bypaper.setdefault(pid,[]).append(inf['question_id'])
        answers=r['evaluation']['answers'];text=' '.join(map(str,answers)).strip().lower()
        kind='yes/no' if text in ('yes','no') else 'extractive' if len(text.split())<=8 else 'free-form'
        difficulty='base-correct' if answer_f1(inf['initial_draft'],r['evaluation'])>=1-1e-8 else 'base-incorrect'
        strata.setdefault((kind,difficulty),set()).add(pid)
    rng=np.random.default_rng(cfg['seed']);queues={k:rng.permutation(sorted(v)).tolist() for k,v in sorted(strata.items())}
    papers=[];qids=[]
    while len(qids)<cfg['screen_queries'] and any(queues.values()):
        for key,queue in queues.items():
            while queue and queue[-1] in papers:queue.pop()
            if not queue:continue
            pid=queue.pop();papers.append(pid);qids.extend(sorted(bypaper[pid]))
            if len(qids)>=cfg['screen_queries']:break
    value=dict(question_ids=qids,paper_ids=papers,fit_membership_hash=source,seed=cfg['seed'],
        selection_split='fit',selection='whole papers; seeded type/base-correctness strata; nominal 48 queries; paper overshoot allowed',
        strata_query_ids={str(k):sorted(q for p in v for q in bypaper[p] if q in qids) for k,v in strata.items()})
    branch_guard();atomic_json(path,value);return value


def evidence_inputs(records,ids):
    """No W9-D/E freeze exists yet: fixed original semantic-HEAVY chunk from FIT.

    Query/draft extraction is deterministic, copied sentence selection using the
    existing W9 sentence splitter. Identical texts are shared by every model.
    PRE is explicitly a textual interface screen, not the later learned method.
    """
    from .chunking import sentences
    output=[]
    for r in records:
        inf=r['inference'];q=inf['question_id']
        if q not in ids:continue
        idx=inf['compressed_indices'][0];chunk=inf['passages'][idx];text=chunk['text']
        # Bound source by whole sentences before any model-specific tokenization.
        spans=sentences(text);parts=[text[a:b] for a,b in spans]
        if not parts:parts=[text]
        source=' '.join(parts[:6])
        words=set(inf['question'].lower().split());draftwords=words|set(inf['initial_draft'].lower().split())
        pick=lambda w:' '.join(sorted(parts[:6],key=lambda s:-len(set(s.lower().split())&w))[:2])
        output.append(dict(question_id=q,paper_id=inf['paper_id'],question=inf['question'],draft=inf['initial_draft'],
            chunk_id=chunk['passage_id'],source=source,query_evidence=pick(words),draft_evidence=pick(draftwords),
            evidence_id=digest([chunk['passage_id'],source,pick(words),pick(draftwords)])))
    return output


def tasks(item,stage):
    from .revision import revision_messages
    q=item['question'];src=item['source'];e=item['draft_evidence'];inf=dict(question=q,initial_draft=item['draft'])
    extract='Extract only source-grounded evidence. Return a Markdown table with columns support_span, source_chunk_id. Quotes must be copied verbatim.\n'
    context=f"\nQuestion: {q}\nSource ID: {item['chunk_id']}\nSource:\n{src}"
    if stage=='smoke':
        return [('T0',[{'role':'user','content':f'Answer concisely from the source. {context}'}]),
            ('T1',[{'role':'user','content':extract+context}]),
            ('T2',[{'role':'user','content':extract+context+'\nInitial draft: '+item['draft']}]),
            ('T3',revision_messages(inf,e,'F0')),('T4',revision_messages(inf,e,'F2'))]
    return [('P0',revision_messages(inf,src,'F0')),('P1',revision_messages(inf,item['query_evidence'],'F0')),
        ('P2',revision_messages(inf,e,'F0')),('P3',revision_messages(inf,e,'F0')),('P4',revision_messages(inf,e,'F2'))]


def run_tests(cfg,models,stage,*,resume=False,freeze=False,smoke_limit=None):
    import torch
    branch_guard();torch.set_num_threads(4)
    records,fit_paths=load_records(load_config(resolve(cfg['fit_config'])),'fit')
    fit_sources={str(path):file_hash(path) for name,path in fit_paths.items() if name not in ('sfr','raw')}
    fit_sources[str(resolve(cfg['fit_config']))]=file_hash(resolve(cfg['fit_config']))
    subset=freeze_subset(cfg,records)
    ids=set(subset['question_ids'][:(smoke_limit or cfg['smoke_queries'])] if stage=='smoke' else subset['question_ids'])
    items=evidence_inputs(records,ids);byid={r['inference']['question_id']:r for r in records}
    # Count the canonical first pass only for interfaces that actually use its draft.
    from .backend import load_tokenizer
    from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
    canonical_tokenizer=load_tokenizer(load_config(resolve(cfg['fit_config'])))
    draft_costs={}
    for item in items:
        rec=byid[item['question_id']];inf=rec['inference']
        prefix,_,_,_=build_mixed_fidelity_sara_prompt(inf['question'],inf['passages'],inf['selected_full_indices'],[],inf['compressed_indices'],{},canonical_tokenizer)
        draft_costs[item['question_id']]=(len(canonical_tokenizer.encode(prefix)),len(rec['canonical']['saved_generated_token_ids']))
    output=resolve(cfg['output_root']);input_path=output/'textual_fit_evidence.json'
    # Immutable shared evidence: screen and smoke use the same full subset snapshot.
    evidence=evidence_inputs(records,set(subset['question_ids']))
    if input_path.exists() and json.loads(input_path.read_text())!=evidence:raise ValueError('Frozen PRE evidence changed')
    atomic_json(input_path,evidence)
    atomic_json(output/f'{stage}_run_config.json',cfg)
    manifests=json.loads((output/'model_cache_manifest.json').read_text())['models']
    cache_rows={r['model_id']:r for r in manifests};status=[];summaries=[]
    for value in models:
        spec=get_model(value);entry=cache_rows.get(spec.model_id,{})
        if entry.get('download_status')!='COMPLETE':
            status.append(dict(model_id=spec.model_id,status='NOT_CACHED',reason=entry.get('download_status','no manifest')));continue
        successful=False
        for precision in cfg['precision_order']:
            runtime=TextRuntime(spec.model_id,entry['revision'],precision,cache_dir=cfg.get('cache_dir'),thinking=cfg['thinking'],max_prompt_tokens=cfg['max_prompt_tokens'])
            identity=dict(fit_sources=fit_sources,model=runtime.identity,model_spec=spec.to_dict(),registry_hash=file_hash(Path(__file__).with_name('model_registry.py')),evaluation_hash=file_hash(ROOT/'src/week6/evaluation.py'),config_hash=digest(cfg),evidence_hash=digest(evidence),stage=stage,
                prompt_hash=file_hash(Path(__file__).with_name('revision.py')),runtime_hash=file_hash(Path(__file__).with_name('model_runtime.py')),
                source_hash=file_hash(__file__),decoding_hash=digest(dict(do_sample=False,max_new_tokens=cfg['max_new_tokens'])),query_ids=sorted(ids))
            folder=output/stage/spec.key/precision;folder.mkdir(parents=True,exist_ok=True)
            try:
                with RowCache(folder/'private_raw_outputs.jsonl',identity,resume=resume) as cache:
                    # Never expose raw reasoning in publication-facing artifacts.
                    (folder/'private_raw_outputs.jsonl').chmod(0o600)
                    pending=[(item,t,msg) for item in items for t,msg in tasks(item,stage)]
                    if any(cache.key(i['question_id'],i['paper_id'],[i['evidence_id'],t],runtime.identity) not in cache.rows for i,t,m in pending):
                        with runtime:
                            for item,task,messages in pending:
                                key=cache.key(item['question_id'],item['paper_id'],[item['evidence_id'],task],runtime.identity)
                                if key in cache.rows:continue
                                isext=task in ('T1','T2');prompt=runtime.render(messages)
                                result=runtime.generate(prompt,cfg['extraction_max_tokens'] if isext else cfg['max_new_tokens'])
                                rec=byid[item['question_id']];f1=answer_f1(result['content'],rec['evaluation']) if not isext else None
                                from src.week6.evaluation import compute_qa_answer_metrics
                                em=compute_qa_answer_metrics(result['content'],rec['evaluation']['answers'],rec['evaluation']['answers_reformatted'])['exact_match'] if not isext else None
                                audit=audit_extraction(result['content'],[{'chunk_id':item['chunk_id'],'text':item['source']}]) if isext else None
                                uses_draft=task in ('T2','T3','T4','P2','P3','P4')
                                dp,dg=draft_costs[item['question_id']] if uses_draft else (0,0)
                                cache.put(key,dict(result,total_query_tokens=result['total_tokens']+dp+dg,
                                    initial_draft_prompt_tokens=dp,initial_draft_generated_tokens=dg,num_LM_calls=1+int(uses_draft),
                                    extraction_tokens=result['total_tokens'] if isext else 0,revision_tokens=0 if isext else result['total_tokens'],
                                    end_to_end_latency=None,cost_note='Text generation latency measured; canonical draft tokens counted when needed. Fixed rule-based PRE evidence has zero extraction LM calls. No deployment end-to-end latency.',
                                    question_id=item['question_id'],paper_id=item['paper_id'],chunk_id=item['chunk_id'],
                                    evidence_id=item['evidence_id'],task=task,revision_mode={'P3':'F0','P4':'F2','T4':'F2'}.get(task,'F0'),
                                    prompt_hash=digest(prompt),decoding_hash=identity['decoding_hash'],config_hash=identity['config_hash'],
                                    evidence_utilization_proxy=bool((set(result['content'].lower().split())-set(item['draft'].lower().split())) & set(item['draft_evidence'].lower().split())) if not isext else None,
                                    f1=f1,exact_match=em,base_f1=answer_f1(rec['inference']['initial_draft'],rec['evaluation']),audit=audit))
                    rows=list(cache.rows.values())
                public=[{k:v for k,v in r.items() if k not in ('raw_output','generated_token_ids','cache_key','row_content_hash','cache_fingerprint')} for r in rows]
                # Public contains only separately parsed final answers, never reasoning.
                atomic_json(folder/'public_results.json',public)
                for task in sorted({r['task'] for r in rows}):
                    group=[r for r in rows if r['task']==task];scored=[r for r in group if r['f1'] is not None]
                    summaries.append(dict(model_id=spec.model_id,revision=entry['revision'],precision=precision,task=task,queries=len(group),
                        f1=float(np.mean([r['f1'] for r in scored])) if scored else None,
                        exact_match=float(np.mean([r['exact_match'] for r in scored])) if scored else None,
                        help_rate=float(np.mean([r['f1']-r['base_f1']>.01 for r in scored])) if scored else None,
                        hurt_rate=float(np.mean([r['f1']-r['base_f1']<-.01 for r in scored])) if scored else None,
                        evidence_utilization_rate=float(np.mean([r['evidence_utilization_proxy'] for r in scored])) if scored else None,faithfulness=float(np.mean([r['audit']['exact_span_support_rate'] for r in group if r.get('audit') and r['audit']['exact_span_support_rate'] is not None])) if any(r.get('audit') and r['audit']['exact_span_support_rate'] is not None for r in group) else None,faithfulness_note='Answer entailment unmeasured; inspect source/final answers manually',
                        empty_rate=float(np.mean([r['empty_output'] for r in group])),malformed_rate=float(np.mean([r['audit']['invalid_format'] if r['audit'] else False for r in group])),
                        mean_input_tokens=float(np.mean([r['prompt_tokens'] for r in group])),mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in group])),
                        mean_total_query_tokens=float(np.mean([r['total_query_tokens'] for r in group])),
                        mean_latency=float(np.mean([r['latency'] for r in group])),p50_latency=float(np.quantile([r['latency'] for r in group],.5)),p95_latency=float(np.quantile([r['latency'] for r in group],.95)),
                        peak_memory_allocated=max(r['peak_memory_allocated'] for r in group),peak_memory_reserved=max(r['peak_memory_reserved'] for r in group),
                        mean_input_characters=float(np.mean([r['input_characters'] for r in group])),vram_status='PASS',quantization=group[0]['quantization']))
                source_byid={i['question_id']:i for i in items}
                atomic_json(folder/'qualitative_sample.json',[dict(r,shared_source=source_byid[r['question_id']]) for r in public[:min(10,len(public))]])
                status.append(dict(model_id=spec.model_id,status='PASS',precision=precision,peak_memory_allocated=max(r['peak_memory_allocated'] for r in rows)))
                successful=True;break
            except (RuntimeError,MemoryError,ImportError,OSError,ValueError) as exc:
                runtime.close()
                status.append(dict(model_id=spec.model_id,status='FAIL',precision=precision,
                    peak_memory_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                    peak_memory_reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
                    dtype=runtime.identity['dtype'],quantization=runtime.identity['quantization'],prompt_tokens=None,generated_tokens=None,
                    reason=type(exc).__name__+': '+str(exc)[:350]))
        if not successful:print('No feasible runtime:',spec.model_id,flush=True)
    selected={get_model(m).model_id for m in models}
    status_path=output/f'{stage}_status.json';summary_path=output/f'{stage}_summary.json'
    prior_status=json.loads(status_path.read_text()) if status_path.exists() else []
    prior_summary=json.loads(summary_path.read_text()) if summary_path.exists() else []
    status=[r for r in prior_status if r['model_id'] not in selected]+status
    summaries=[r for r in prior_summary if r['model_id'] not in selected]+summaries
    atomic_json(status_path,status)
    atomic_json(summary_path,summaries)
    if freeze:
        if stage!='screen':raise ValueError('Only complete FIT screen can freeze a model panel')
        freeze_panel(cfg,summaries,models,subset)
    return status


def freeze_panel(cfg,summaries,models,subset):
    from .model_registry import runtime_identity
    # P2/P3/P4 share fixed draft-aware evidence; aggregate before the ONE global tolerance.
    required={get_model(x).model_id for x in models};options=[]
    for model in required:
        rows=[r for r in summaries if r['model_id']==model and r['task'] in ('P2','P3','P4')]
        if len(rows)!=3 or any(r['queries']!=len(subset['question_ids']) for r in rows):raise ValueError('Incomplete model screen; cannot freeze panel')
        if any(r['empty_rate']>0.05 or r['malformed_rate']>0.05 for r in rows):continue
        options.append(dict(model_id=model,quality=float(np.mean([r['f1'] for r in rows])),
            tokens=float(np.mean([r['mean_total_query_tokens'] for r in rows])),latency=float(np.mean([r['mean_latency'] for r in rows])),
            memory=max(r['peak_memory_allocated'] for r in rows),revision=rows[0]['revision'],precision=rows[0]['precision']))
    canonical=next((r for r in options if r['model_id']==CANONICAL),None)
    if canonical is None:raise ValueError('Canonical Mistral must pass the complete screen')
    # Explicit manual support review is required because answer entailment is unmeasured.
    review_path=resolve(cfg['output_root'])/'qualitative_review.json'
    if not review_path.exists():raise FileNotFoundError('Review fixed qualitative samples; provide qualitative_review.json with approved_model_ids and evidence_hash')
    review=json.loads(review_path.read_text())
    if review.get('evidence_hash')!=digest(json.loads((resolve(cfg['output_root'])/'textual_fit_evidence.json').read_text())):raise ValueError('Qualitative review is for different evidence')
    approved=set(review['approved_model_ids']);alts=[r for r in options if r['model_id']!=CANONICAL and r['model_id'] in approved]
    if alts:
        best=max(r['quality'] for r in alts)
        alts=sorted(alts,key=lambda r:(0 if r['quality']>=best-cfg['quality_tolerance'] else 1,
            r['tokens'] if r['quality']>=best-cfg['quality_tolerance'] else -r['quality'],r['latency'],r['memory'],get_model(r['model_id']).reasoning))[:min(cfg['max_alternatives'],2)]
    panel=[canonical,*alts]
    value=dict(frozen=True,selection_split='fit',question_ids=subset['question_ids'],model_panel=panel,
        first_pass_sara_model=CANONICAL,no_confirmation_selection=True,qualitative_review_hash=file_hash(review_path))
    value['panel_hash']=digest(value)
    path=resolve(cfg['output_root'])/'frozen_model_panel.json'
    if path.exists() and json.loads(path.read_text())!=value:raise FileExistsError('Refusing to replace a frozen model panel')
    branch_guard();atomic_json(path,value)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=['cache','smoke','screen'])
    p.add_argument('--config',default='configs/week9/w9pre_models.yaml');p.add_argument('--model',action='append')
    p.add_argument('--local-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--smoke-limit',type=int,choices=[1,2,3])
    a=p.parse_args(argv);branch_guard();cfg=yaml.safe_load(resolve(a.config).read_text());models=a.model or cfg['models']
    if cfg['first_pass_sara_model']!=CANONICAL:raise ValueError('First-pass SARA must remain the trained Mistral checkpoint')
    if a.stage=='cache':
        if a.freeze:raise ValueError('Cache cannot select a model')
        return cache_models(cfg,models,a.local_only)
    return run_tests(cfg,models,a.stage,resume=a.resume,freeze=a.freeze,smoke_limit=a.smoke_limit)
