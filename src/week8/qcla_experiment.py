"""Measured W8-F experiment, input provenance, resumable conditioning and checkpoints."""
from __future__ import annotations
import gc, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from src.week8.data import PROJECT_ROOT,load_config,load_week8_split,assert_split_disjointness
from src.week8.utility_training import load_surface
from src.week8.generator_features import sha256_file
from src.week8.qcla import build_qcla_adapter
from src.week8.qcla_training import QCLATrainingDataset,single_query_collate,train_qcla_epoch,evaluate_qcla

MODEL='Salesforce/SFR-Embedding-Mistral'
SCHEMA='w8f_canonical_qcla_v2'
DEFAULT_OUT=PROJECT_ROOT/'results/week8/w8f_qcla'


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def atomic_torch(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def input_sources(settings):
    cfg=load_config();ckpt=PROJECT_ROOT/cfg['sara']['checkpoint_path']
    snapshot=Path(snapshot_download(MODEL,local_files_only=True))
    names=['src/week8/qcla.py','src/week8/qcla_training.py','src/week8/qcla_experiment.py',
        'scripts/week8/run_w8f_qcla_experiments.py','src/week8/generator_features.py',
        'src/week8/utility_training.py','src/week7/w7e/prompt_builder.py','src/week6/evaluation.py',
        cfg['splits']['train_retrieval'],cfg['splits']['train_sfr_embeddings'],
        cfg['splits']['fit_manifest'],cfg['splits']['internal_val_manifest'],
        'results/week8/surfaces/promotion_surface_fit.jsonl','results/week8/surfaces/promotion_surface_internal_val.jsonl',
        'results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl',
        'Baselines/SARA-main/src/model/loader.py','Baselines/SARA-main/src/model/xMistral/modeling_xmistral.py']
    paths=[PROJECT_ROOT/n for n in names]
    paths += sorted(p for p in ckpt.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.model','.jinja'))
    hashes={str(p.relative_to(PROJECT_ROOT)):sha256_file(p) for p in paths}
    metadata={'schema':SCHEMA,'sources':hashes,'settings':settings,'conditioning_model':MODEL,
        'conditioning_snapshot':snapshot.name,'conditioning_max_tokens':512,
        'query_format':'Instruct: Given a scientific question, retrieve passages that answer the question.\nQuery: {question}',
        'draft_format':'raw frozen unadapted canonical k4 answer; empty text is encoded, never zero-filled',
        'normalization':'SFR last-token pooling, normalize_embeddings=False; same encoder family as passage cache',
        'training_target':'first gold reference, up to 127 content tokens plus EOS; query-weighted NLL',
        'fit_only_training':True,'epochs_fixed_before_validation':True,'validation_status':'previously inspected development confirmation; no TEST',
        'QCLA2_draft_source':'frozen unadapted k4, never adapted draft or gold',
        'baseline':'identity QCLA-0 on exact canonical semantic k4 allocation',
        'cost_note':'QCLA-2 requires original k4 generation plus adapted regeneration and draft encoding; conditioning costs reported separately, not claimed as deployed end-to-end latency'}
    fingerprint=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    return fingerprint,metadata,snapshot


def load_inputs(limit=None):
    data={}
    canonical_path=PROJECT_ROOT/'results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl'
    canonical={}
    for line in canonical_path.read_text().splitlines():
        row=json.loads(line)
        if int(row['k'])==4:canonical[str(row['question_id'])]=row
    for split in ['fit','internal_val']:
        records,sfr,_=load_week8_split(split)
        frame=load_surface(split)
        byid={str(r['question_id']):r for r in records}
        groups={str(q):g for q,g in frame.groupby('question_id',sort=True)}
        if set(byid)!=set(groups):raise ValueError('Canonical routing does not cover split')
        for q,g in groups.items():
            if q not in canonical or not np.isfinite(canonical[q]['latency']):raise ValueError('Missing measured baseline generation latency')
            g['base_generation_latency']=float(canonical[q]['latency'])
        qids=sorted(groups)[:limit] if limit else sorted(groups)
        data[split]={'records':[byid[q] for q in qids],'sfr':sfr,'routing':{q:groups[q] for q in qids}}
    assert_split_disjointness(data['fit']['records'],data['internal_val']['records'])
    return data


def conditioning_inputs(data,folder,snapshot,device):
    # Per-query text hashes allow a smoke to fill the same measured cache as a
    # later full run, while source guards reject changed encoder/data definitions.
    folder.mkdir(parents=True,exist_ok=True)
    encoder_sources={'snapshot':snapshot.name,'model':MODEL,'max_tokens':512,'normalize':False,
        'definition_sha256':hashlib.sha256((Path(__file__).read_text()).encode()).hexdigest()}
    guard=folder/'sources.json'
    if guard.exists() and json.loads(guard.read_text())!=encoder_sources:raise ValueError('Conditioning definition changed; use a fresh cache')
    atomic_json(guard,encoder_sources)
    values={};pending=[]
    for split,d in data.items():
        values[split]={'query':{},'draft':{},'timing':{}}
        for r in d['records']:
            q=str(r['question_id']);draft=str(d['routing'][q].iloc[0].draft_answer)
            text='Instruct: Given a scientific question, retrieve passages that answer the question.\nQuery: '+str(r['question']).strip()
            source=hashlib.sha256(json.dumps([split,q,str(r['paper_id']),text,draft,encoder_sources],sort_keys=True).encode()).hexdigest()
            p=folder/f'{split}_{q}.pt'
            if p.exists():
                row=torch.load(p,map_location='cpu',weights_only=False)
                if row['source']!=source:raise ValueError(f'{q}: stale conditioning vectors')
                for kind in ['query','draft']:
                    v=row[kind]
                    if v.shape!=(4096,) or not torch.isfinite(v).all() or not v.norm()>0:raise ValueError('Invalid measured conditioning vector')
                    values[split][kind][q]=v
                values[split]['timing'][q]=row['seconds']
            else:pending.append((split,q,text,draft,source,p))
    if pending:
        from sentence_transformers import SentenceTransformer
        if str(device).startswith('cuda') and torch.cuda.mem_get_info()[0]<18*1024**3:raise RuntimeError('Need at least 18 GiB free GPU memory')
        encoder=SentenceTransformer(str(snapshot),device=device,local_files_only=True,model_kwargs={'torch_dtype':torch.bfloat16})
        encoder.max_seq_length=512
        try:
            for start in tqdm(range(0,len(pending),4),desc='Measured SFR query/draft conditioning'):
                batch=pending[start:start+4];texts=[x[2] for x in batch]+[x[3] for x in batch]
                begin=time.perf_counter()
                with torch.inference_mode():
                    vectors=encoder.encode(texts,batch_size=4,convert_to_tensor=True,normalize_embeddings=False,show_progress_bar=False).detach().float().cpu()
                seconds=(time.perf_counter()-begin)/len(texts)
                for i,(split,q,_,_,source,p) in enumerate(batch):
                    row={'source':source,'query':vectors[i],'draft':vectors[i+len(batch)],'seconds':seconds,'measured':True}
                    if not all(v.shape==(4096,) and torch.isfinite(v).all() and v.norm()>0 for v in [row['query'],row['draft']]):raise ValueError('Invalid measured SFR vector')
                    atomic_torch(p,row)
                    for kind in ['query','draft']:values[split][kind][q]=row[kind]
                    values[split]['timing'][q]=seconds
        finally:
            del encoder;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    return values


def loader(dataset,seed=None):
    generator=torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset,batch_size=1,shuffle=seed is not None,generator=generator,collate_fn=single_query_collate)


def run_experiment(output_dir=DEFAULT_OUT,*,smoke=False,device='cuda',epochs=2,prepare_only=False,check=False):
    import fcntl
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another W8-F worker owns this run') from exc
        return _run(output_dir,smoke,device,epochs,prepare_only,check)


def _run(output_dir,smoke,device,epochs,prepare_only,check):
    if epochs<1:raise ValueError('Epoch count must be positive')
    torch.set_num_threads(4)
    settings={'seed':42,'epochs':1 if smoke else epochs,'hidden_dim':2048,'learning_rate':1e-4,
        'gradient_accumulation_steps':2 if smoke else 16,'smoke':smoke,'limit':2 if smoke else None,
        'variants':['qcla_0','qcla_1','qcla_2'],'device':device}
    fingerprint,metadata,snapshot=input_sources(settings)
    run=output_dir/('smoke' if smoke else 'full');run.mkdir(exist_ok=True)
    if check:
        data=load_inputs(settings['limit'])
        counts={k:len(d['records']) for k,d in data.items()}
        print('Verified canonical inputs and split disjointness:',counts,flush=True)
        return counts
    provenance=run/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Frozen W8-F run inputs/settings changed; inspect provenance and use a new output directory')
    atomic_json(provenance,{'fingerprint':fingerprint,**metadata})
    summary_path=run/'summary.json'
    if summary_path.exists():
        saved=json.loads(summary_path.read_text())
        for name,digest in saved['artifact_sha256'].items():
            if sha256_file(run/name)!=digest:raise ValueError(f'Completed artifact changed: {name}')
        print('Verified completed measured W8-F run:',summary_path,flush=True);return saved
    data=load_inputs(settings['limit'])
    conditioning=conditioning_inputs(data,output_dir/'conditioning',snapshot,device)
    datasets={s:QCLATrainingDataset(d['records'],d['sfr'],conditioning[s]['query'],conditioning[s]['draft'],routing_by_qid=d['routing']) for s,d in data.items()}
    if prepare_only:
        print('Measured conditioning inputs complete',flush=True);return {s:len(ds) for s,ds in datasets.items()}
    cfg=load_config()
    import src
    baseline=str(PROJECT_ROOT/'Baselines/SARA-main/src')
    if baseline not in src.__path__:src.__path__.append(baseline)
    from src.model.loader import load_sara_for_eval
    if str(device).startswith('cuda') and torch.cuda.mem_get_info()[0]<18*1024**3:raise RuntimeError('Need at least 18 GiB free GPU memory')
    atomic_json(run/'protocol_frozen_before_validation.json',{'fingerprint':fingerprint,**settings,
        'selection':'Fixed epoch budget; evaluate all three variants. No best-validation checkpoint selection.'})
    model,tokenizer=load_sara_for_eval(str(PROJECT_ROOT/cfg['sara']['checkpoint_path']),device=device)
    results=[];artifacts={}
    try:
        for variant in settings['variants']:
            seed_all(settings['seed']);adapter=build_qcla_adapter(variant,hidden_dim=settings['hidden_dim']).to(device)
            ckpt=run/f'{variant}.pt';history=[]
            if variant!='qcla_0':
                optimizer=torch.optim.AdamW(adapter.parameters(),lr=settings['learning_rate'])
                completed=0
                if ckpt.exists():
                    saved=torch.load(ckpt,map_location=device,weights_only=False)
                    if saved['fingerprint']!=fingerprint:raise ValueError('Checkpoint source mismatch')
                    adapter.load_state_dict(saved['adapter_state']);optimizer.load_state_dict(saved['optimizer_state'])
                    completed=saved['completed_epochs'];history=saved['history']
                for epoch in range(completed,settings['epochs']):
                    begin=time.perf_counter()
                    loss=train_qcla_epoch(adapter,model,tokenizer,tqdm(loader(datasets['fit'],42+epoch),desc=f'{variant} FIT epoch {epoch+1}'),optimizer,
                        gradient_accumulation_steps=settings['gradient_accumulation_steps'],device=device)
                    history.append({'epoch':epoch+1,'fit_query_mean_answer_nll':loss,'seconds':time.perf_counter()-begin})
                    atomic_torch(ckpt,{'fingerprint':fingerprint,'variant':variant,'adapter_state':adapter.state_dict(),
                        'optimizer_state':optimizer.state_dict(),'completed_epochs':epoch+1,'history':history,'settings':settings})
                    print(variant,'epoch',epoch+1,'measured FIT NLL',loss,flush=True)
                del optimizer
            elif not ckpt.exists():
                atomic_torch(ckpt,{'fingerprint':fingerprint,'variant':variant,'adapter_state':adapter.state_dict(),'completed_epochs':0,'settings':settings})
            if variant!='qcla_0' and not torch.count_nonzero(adapter.mlp_f[-1].weight.detach()).item():
                raise ValueError('Adapter remained an untrained identity after FIT training')
            # Evaluation rows are resumable only after training is finished and the checkpoint is frozen.
            checkpoint_hash=sha256_file(ckpt);prediction_path=run/f'{variant}_internal_val.jsonl';rows=[]
            if prediction_path.exists():
                rows=[json.loads(l) for l in prediction_path.read_text().splitlines()]
                if any(r['checkpoint_sha256']!=checkpoint_hash for r in rows):raise ValueError('Predictions use a different adapter checkpoint')
            done={r['question_id'] for r in rows}
            wanted={str(r['question_id']) for r in data['internal_val']['records']}
            if len(done)!=len(rows) or not done<=wanted:raise ValueError('Duplicate or extraneous prediction rows')
            with prediction_path.open('a') as f:
                for batch in tqdm(loader(datasets['internal_val']),desc=f'{variant} measured generation'):
                    if batch['question_id'] in done:continue
                    row=evaluate_qcla(adapter,model,tokenizer,[batch],device=device,generate=True)[0]
                    if variant=='qcla_0' and row['prediction']!=row['baseline_draft'].strip():
                        raise ValueError(f"{row['question_id']}: identity baseline differs from canonical k4; do not overwrite saved answers")
                    # QCLA-2 additionally incurs generation of the original draft, before conditioning.
                    g=data['internal_val']['routing'][row['question_id']].iloc[0]
                    base_lat=float(g['base_generation_latency'])
                    row.update(checkpoint_sha256=checkpoint_hash,variant=variant,conditioning_encode_seconds=conditioning['internal_val']['timing'][row['question_id']]*(2 if variant=='qcla_2' else (1 if variant=='qcla_1' else 0)),
                        cumulative_prompt_tokens=row['prompt_tokens']*(2 if variant=='qcla_2' else 1),
                        generation_passes=2 if variant=='qcla_2' else 1,
                        baseline_draft_generation_latency_sec=base_lat if variant=='qcla_2' else 0.)
                    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();rows.append(row)
            frame=pd.DataFrame(rows)
            results.append({'variant':variant,'validation_queries':len(rows),'trainable_params':sum(p.numel() for p in adapter.parameters() if p.requires_grad),
                'validation_f1':float(frame.token_f1.mean()),'validation_query_mean_answer_nll':float(frame.answer_nll.mean()),
                'mean_final_prompt_tokens':float(frame.prompt_tokens.mean()),'mean_cumulative_prompt_tokens':float(frame.cumulative_prompt_tokens.mean()),
                'mean_adapted_generation_latency_sec':float(frame.generation_latency_sec.mean()),
                'mean_estimated_cumulative_generation_latency_sec':float((frame.generation_latency_sec+frame.baseline_draft_generation_latency_sec).mean()),
                'mean_conditioning_encode_seconds':float(frame.conditioning_encode_seconds.mean()),
                'history':history,'end_to_end_latency_sec':None})
            for p in [ckpt,prediction_path]:artifacts[p.name]=sha256_file(p)
            del adapter;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        summary={'fingerprint':fingerprint,'smoke_only':smoke,'settings':settings,'results':results,
            'artifact_sha256':artifacts,'baseline_canonical_match':True,
            'validation_status':metadata['validation_status'],'no_validation_selection':True,
            'latency_note':'Measured adapted generation and batched conditioning encode time, not end-to-end deployment latency. QCLA-2 also requires the original k4 pass; output records retain its cached generation latency.',
            'TEST_used':False}
        atomic_json(summary_path,summary)
        pd.DataFrame([{k:v for k,v in r.items() if k!='history'} for r in results]).to_csv(run/'comparison.csv',index=False)
        print(pd.read_csv(run/'comparison.csv').to_string(index=False),flush=True)
        return summary
    finally:
        del model,tokenizer;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
