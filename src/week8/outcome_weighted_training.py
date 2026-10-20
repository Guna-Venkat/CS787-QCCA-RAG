"""Measured W8-G: independent uniform/HC/OW LoRA runs on FIT only.

Gamma and epoch budget are fixed before validation. Two non-oracle evaluation
contexts: adapted semantic k4, and revision under the frozen W8-E policy with
its original unadapted k4 draft. No validation loss/outcome selects actions.
"""
from __future__ import annotations
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from src.week8.data import PROJECT_ROOT,load_config
from src.week8.generator_features import sha256_file,canonical_k4_context
from src.week8.qcla_experiment import load_inputs,atomic_json,atomic_torch,seed_all
from src.week8.qcla_training import gold_target_inputs,answer_loss,single_query_collate
from src.week8.hard_case_dataset import build_hard_case_dataset,HardCasePromptDataset,categorize_hard_cases,CASE_TYPES
from src.week8.hard_case_lora import setup_lora_generator,adapter_state,load_adapter_state,train_generator_lora_epoch
from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
from src.week6.evaluation import compute_qa_answer_metrics
from src.week8.selective_controller import controller_sources

OUT=PROJECT_ROOT/'results/week8/w8g_lora'
VARIANTS=['original_SARA','uniform_SFT','HC_LoRA','OW_LoRA']


def frozen_policy():
    folder=PROJECT_ROOT/'results/week8/w8e_controller'
    summary=json.loads((folder/'controller_summary.json').read_text())
    if summary['source_hashes']!=controller_sources():raise ValueError('Frozen W8-E sources changed')
    path=folder/f"{summary['selected_policy']}_internal_val_decisions.csv"
    keys=['question_id','paper_id','selected_action','candidate_passage_index','semantic_rank','bm25_rank']
    decisions=pd.read_csv(path,dtype={'question_id':str,'paper_id':str},usecols=keys)
    if decisions.question_id.duplicated().any() or len(decisions)!=342:raise ValueError('Incomplete frozen validation policy')
    if not decisions.selected_action.isin(['STOP','PROMOTE']).all():raise ValueError('Unknown frozen action')
    if summary['oracle_gate'] or summary['max_promotions']!=1:raise ValueError('Only the real frozen single-promotion policy is supported')
    return summary['selected_policy'],decisions.set_index('question_id'),path


def sources(settings,policy_path):
    cfg=load_config();ckpt=PROJECT_ROOT/cfg['sara']['checkpoint_path']
    names=['src/week8/hard_case_dataset.py','src/week8/hard_case_lora.py','src/week8/outcome_weighted_training.py',
        'scripts/week8/run_w8g_lora_training.py','src/week8/qcla_training.py','src/week8/qcla_experiment.py',
        'src/week8/generator_features.py','src/week8/utility_training.py','src/week7/w7e/prompt_builder.py',
        'src/week6/evaluation.py','Baselines/SARA-main/src/model/loader.py',
        'Baselines/SARA-main/src/model/xMistral/modeling_xmistral.py',
        cfg['splits']['train_retrieval'],cfg['splits']['train_sfr_embeddings'],cfg['splits']['fit_manifest'],cfg['splits']['internal_val_manifest'],
        'results/week8/surfaces/promotion_surface_fit.jsonl','results/week8/surfaces/promotion_surface_internal_val.jsonl',
        'results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl',
        'results/week8/w8e_controller/controller_summary.json','results/week8/w8e_controller/frozen_controller.json']
    paths=[PROJECT_ROOT/n for n in names]+[policy_path]
    paths+=sorted(p for p in ckpt.iterdir() if p.is_file() and p.suffix in ['.json','.safetensors','.model','.jinja'])
    meta={'schema':'w8g_real_additive_lora_v1','settings':settings,
        'source_sha256':{str(p.relative_to(PROJECT_ROOT)):sha256_file(p) for p in paths},
        'environment':{'torch':str(torch.__version__),'transformers':transformers.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        'training':'All FIT queries; identical outcome-derived TRAINING contexts in uniform/HC/OW. Only training weights differ. Raw weights normalized by the full FIT mean per variant.',
        'targets':'First gold reference, up to 127 content tokens plus EOS; generation-prefix token IDs preserved',
        'inference':'k4 adapts direct generation; frozen_controller adapts promoted revisions only and reuses original k4 draft on STOP',
        'LoRA':'New additive factors on original q/k/v/o modules. Original trained SARA PEFT adapter, base model and projector stay frozen/unmerged.',
        'selection':'Fixed gamma=2, fixed epochs and all named variants. No validation or TEST selection.',
        'validation_status':'Previously inspected INTERNAL-VAL development confirmation. TEST remains unused.'}
    return hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),meta


def choose_smoke(data,decisions):
    fit=data['fit'];groups=fit['routing']
    categories=categorize_hard_cases({q:g.to_dict('records') for q,g in groups.items()})
    selected=[]
    for kind in CASE_TYPES:
        candidates=sorted(q for q,c in categories.items() if c['case_type']==kind)
        if not candidates:raise ValueError(f'No real FIT smoke case for {kind}')
        selected.append(candidates[0])
    vals=[]
    for action in ['STOP','PROMOTE']:
        candidates=sorted(q for q in data['internal_val']['routing'] if decisions.loc[q,'selected_action']==action)
        if not candidates:raise ValueError(f'No real validation {action} smoke case')
        vals.append(candidates[0])
    for split,qids in [('fit',selected),('internal_val',vals)]:
        part=data[split];records={str(r['question_id']):r for r in part['records']}
        part['records']=[records[q] for q in sorted(qids)];part['routing']={q:part['routing'][q] for q in sorted(qids)}
    return data


def evaluation_context(record,group,embeddings,tokenizer,policy_row=None):
    """Build a fixed action's context without reading any F1/delta/outcome label."""
    question,prompt,_,compressed,comp,full=canonical_k4_context(group,record,embeddings,tokenizer)
    selected=None;action='GENERATE_K4'
    if policy_row is not None:
        if str(policy_row.paper_id)!=str(record['paper_id']):raise ValueError('Frozen action paper mismatch')
        action=str(policy_row.selected_action)
        if action=='PROMOTE':
            selected=int(policy_row.candidate_passage_index)
            candidate=group[group.candidate_passage_index.astype(int)==selected]
            if len(candidate)!=1:raise ValueError('Frozen policy selected a noncanonical passage')
            row=candidate.iloc[0]
            if int(row.semantic_rank)!=int(policy_row.semantic_rank) or int(row.bm25_rank)!=int(policy_row.bm25_rank):raise ValueError('Frozen rank identity changed')
            full=sorted(full+[selected]);compressed=sorted(set(range(10))-set(full))
            prompt,_,actual,_=build_mixed_fidelity_sara_prompt(question,record['passages'][:10],full,[],compressed,{},tokenizer,hybrid_medium=False)
            if actual!=compressed:raise ValueError('Compressed order changed')
            comp=embeddings[compressed]
        elif action!='STOP':raise ValueError('Unknown fixed action')
        elif any(pd.notna(policy_row[k]) for k in ['candidate_passage_index','semantic_rank','bm25_rank']):raise ValueError('STOP carries a selected passage')
    return prompt,comp,full,compressed,selected,action


def baseline_row(record,group,embeddings,tokenizer,policy_row=None):
    prompt,_,full,compressed,selected,action=evaluation_context(record,group,embeddings,tokenizer,policy_row)
    first=group.iloc[0];promoted=action=='PROMOTE'
    ref=group[group.candidate_passage_index.astype(int)==selected].iloc[0] if promoted else first
    prediction=str(ref.promoted_prediction) if promoted else str(first.draft_answer)
    latency=float(ref.promotion_latency) if promoted else float(first.base_generation_latency)
    tokens=len(tokenizer(prompt)['input_ids'])
    base_prompt=canonical_k4_context(group,record,embeddings,tokenizer)[1]
    base_tokens=len(tokenizer(base_prompt)['input_ids'])
    is_controller=policy_row is not None
    cumulative=base_tokens+tokens if is_controller and promoted else tokens
    cumulative_latency=float(first.base_generation_latency)+latency if is_controller and promoted else latency
    metrics=compute_qa_answer_metrics(prediction,record['gold_answers'],record.get('gold_answers_reformatted',record['gold_answers']))
    return {'question_id':str(record['question_id']),'paper_id':str(record['paper_id']),
        'context':'frozen_controller' if is_controller else 'k4','selected_action':action,'candidate_passage_index':selected,
        'full_indices':full,'compressed_indices':compressed,'prediction':prediction,'token_f1':metrics['token_f1'],
        'final_prompt_tokens':tokens,'cumulative_prompt_tokens':cumulative,
        'estimated_cumulative_generation_latency_sec':cumulative_latency,'new_generation_latency_sec':None,
        'generation_source':'saved_real_canonical_SARA','answer_nll':None}


@torch.no_grad()
def generate_row(model,tokenizer,record,group,embeddings,policy_row=None):
    reference=baseline_row(record,group,embeddings,tokenizer,policy_row)
    if reference['selected_action']=='STOP':return reference  # original draft is the fixed policy output
    model.eval();device=next(model.parameters()).device
    prompt,comp,_,_,_,_=evaluation_context(record,group,embeddings,tokenizer,policy_row)
    ids,targets=gold_target_inputs(tokenizer,prompt,record,device)
    nll=float(answer_loss(model,ids,targets,comp.to(device=device,dtype=model.dtype)))
    prefix=tokenizer(prompt,return_tensors='pt')['input_ids'].to(device)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    start=time.perf_counter()
    out=model.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),retrieval_embeds=comp.to(device=device,dtype=model.dtype),
        max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    duration=time.perf_counter()-start
    prediction=tokenizer.decode(out[0],skip_special_tokens=True).strip()
    metrics=compute_qa_answer_metrics(prediction,record['gold_answers'],record.get('gold_answers_reformatted',record['gold_answers']))
    cumulative=duration+float(group.iloc[0].base_generation_latency) if policy_row is not None else duration
    return {**reference,'prediction':prediction,'token_f1':metrics['token_f1'],'new_generation_latency_sec':duration,
        'estimated_cumulative_generation_latency_sec':cumulative,'generation_source':'new_measured_W8G_SARA','answer_nll':nll}


def summarize(rows,baseline):
    frame=pd.DataFrame(rows);base=pd.DataFrame(baseline);result={}
    for context,g in frame.groupby('context'):
        ref=base[base.context==context].set_index('question_id')
        delta=g.token_f1.to_numpy()-g.question_id.map(ref.token_f1).to_numpy()
        result[context]={'queries':len(g),'downstream_f1':float(g.token_f1.mean()),'gain_over_original_same_policy':float(delta.mean()),
            'help_rate_vs_original':float((delta>.01).mean()),'hurt_rate_vs_original':float((delta<-.01).mean()),
            'mean_final_prompt_tokens':float(g.final_prompt_tokens.mean()),'mean_cumulative_prompt_tokens':float(g.cumulative_prompt_tokens.mean()),
            'mean_estimated_cumulative_generation_latency_sec':float(g.estimated_cumulative_generation_latency_sec.mean()),
            'generation_promote_rate':float(g.selected_action.eq('PROMOTE').mean()),'end_to_end_latency_sec':None}
    return result


def run_w8g(output_dir=OUT,*,smoke=False,device='cuda',epochs=2,check=False):
    import fcntl
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another worker owns this W8-G output directory') from exc
        return _run(folder,smoke,device,epochs,check)


def _run(folder,smoke,device,epochs,check):
    if epochs<1:raise ValueError('Epoch budget must be positive')
    torch.set_num_threads(4)
    policy_name,decisions,policy_path=frozen_policy()
    settings={'smoke':smoke,'epochs':1 if smoke else epochs,'seed':42,'gamma':2.,'learning_rate':1e-4,
        'gradient_accumulation_steps':3 if smoke else 16,'r':8,'alpha':16,'dropout':.05,
        'target_modules':['q_proj','k_proj','v_proj','o_proj'],'variants':VARIANTS,'frozen_policy':policy_name,'device':device}
    fingerprint,metadata=sources(settings,policy_path)
    run=folder/('smoke' if smoke else 'full')
    if not check and (run/'summary.json').exists():
        old=json.loads((run/'summary.json').read_text())
        if old['fingerprint']!=fingerprint:raise ValueError('Completed W8-G sources/settings changed; use a new experiment directory')
        for name,digest in old['artifact_sha256'].items():
            if sha256_file(run/name)!=digest:raise ValueError(f'Completed W8-G artifact changed: {name}')
        print('Verified completed measured W8-G run:',run,flush=True);return old
    data=load_inputs()
    if set(decisions.index)!=set(data['internal_val']['routing']):raise ValueError('Frozen policy coverage differs from validation')
    if smoke:data=choose_smoke(data,decisions)
    # Missing SFR/routing is checked before spending GPU work.
    for part in data.values():
        for r in part['records']:
            q=str(r['question_id']);v=part['sfr'].get(q)
            if v is None or v.shape!=(10,4096) or not torch.isfinite(v).all():raise ValueError('Missing real passage embeddings')
    fit_rows=[r for g in data['fit']['routing'].values() for r in g.to_dict('records')]
    counts={k:len(v['records']) for k,v in data.items()}
    if check:print('Verified real FIT/VAL surfaces, frozen policy, embeddings and paper disjointness:',counts,flush=True);return counts
    run.mkdir(parents=True,exist_ok=True)
    provenance=run/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())['fingerprint']!=fingerprint:raise ValueError('Resume provenance changed; inspect existing run')
    atomic_json(provenance,{'fingerprint':fingerprint,**metadata})
    atomic_json(run/'protocol_frozen_before_validation.json',{'fingerprint':fingerprint,**settings,
        'fit_queries':counts['fit'],'validation_queries':counts['internal_val'],'no_validation_selection':True})
    import src
    sara_src=str(PROJECT_ROOT/'Baselines/SARA-main/src')
    if sara_src not in src.__path__:src.__path__.append(sara_src)
    from src.model.loader import load_sara_for_eval
    from transformers import AutoTokenizer
    cfg=load_config();checkpoint=PROJECT_ROOT/cfg['sara']['checkpoint_path']
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True)
    baseline=[]
    val=data['internal_val'];records={str(r['question_id']):r for r in val['records']}
    for q,r in records.items():
        for context,action in [('k4',None),('frozen_controller',decisions.loc[q])]:
            baseline.append(baseline_row(r,val['routing'][q],val['sfr'][q],tokenizer,action))
    original_path=run/'original_SARA_predictions.jsonl'
    original_text=''.join(json.dumps(r,allow_nan=False)+'\n' for r in baseline)
    if original_path.exists() and original_path.read_text()!=original_text:raise ValueError('Saved baseline changed')
    original_path.write_text(original_text)
    results={'original_SARA':summarize(baseline,baseline)};artifacts={original_path.name:sha256_file(original_path)};histories={};proofs={}
    for variant,mode in [('uniform_SFT','uniform'),('HC_LoRA','hard_case'),('OW_LoRA','outcome_weighted')]:
        examples,type_counts=build_hard_case_dataset(data['fit']['records'],fit_rows,tokenizer,mode=mode,gamma=settings['gamma'])
        weight_mean=float(np.mean([e['raw_weight'] for e in examples]))
        for e in examples:e['weight']=e['raw_weight']/weight_mean
        manifest=[{k:e[k] for k in ['question_id','paper_id','case_type','baseline_full_indices','full_indices','compressed_indices','training_promoted_passage_index','raw_weight','weight','max_delta','min_delta']} for e in examples]
        atomic_json(run/f'{variant}_fit_examples.json',{'type_counts':type_counts,'raw_weight_mean':weight_mean,'examples':manifest})
        if str(device).startswith('cuda') and torch.cuda.mem_get_info()[0]<18*1024**3:raise RuntimeError('Need at least 18 GiB free GPU memory')
        seed_all(42)
        model,model_tokenizer=load_sara_for_eval(str(checkpoint),device=device)
        trainable=mapped=optimizer=saved=None
        try:
            if model_tokenizer('test')['input_ids']!=tokenizer('test')['input_ids']:raise ValueError('Tokenizer mismatch')
            # Fresh original SARA for EVERY variant; zero residuals preserve its trained PEFT adapter.
            original_parameter_versions={name:p._version for name,p in model.named_parameters()}
            model=setup_lora_generator(model,r=settings['r'],lora_alpha=settings['alpha'],lora_dropout=settings['dropout'])
            expected_projections=4*model.config.num_hidden_layers
            if len(model._w8g_projection_names)!=expected_projections:raise ValueError('Not all q/k/v/o projections received a residual adapter')
            trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
            optimizer=torch.optim.AdamW(trainable.values(),lr=settings['learning_rate'])
            ckpt=run/f'{variant}.pt';history=[];completed=0
            if ckpt.exists():
                saved=torch.load(ckpt,map_location=device,weights_only=False)
                if saved['fingerprint']!=fingerprint:raise ValueError('Adapter checkpoint source mismatch')
                load_adapter_state(model,saved['adapter_state']);optimizer.load_state_dict(saved['optimizer_state'])
                completed=saved['completed_epochs'];history=saved['history'];identity_checks=saved['identity_checks']
            else:
                identity_checks=0
                # Check actual untrained decoding on a STOP and PROMOTE query. In full mode use the same two fixed kinds.
                checks=[]
                for action in ['STOP','PROMOTE']:
                    candidates=sorted(q for q in records if decisions.loc[q,'selected_action']==action)
                    if candidates:checks.append(candidates[0])
                for q in checks:
                    for action in [None,decisions.loc[q]]:
                        actual=generate_row(model,tokenizer,records[q],val['routing'][q],val['sfr'][q],action)
                        reference=baseline_row(records[q],val['routing'][q],val['sfr'][q],tokenizer,action)
                        if actual['prediction'].strip()!=reference['prediction'].strip():raise ValueError('Zero-initialized W8-G residual changed the canonical original SARA answer')
                        identity_checks+=1
            ds=HardCasePromptDataset(examples,tokenizer)
            for epoch in range(completed,settings['epochs']):
                seed_all(42+epoch)
                batches=DataLoader(ds,batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(42+epoch),collate_fn=single_query_collate)
                start=time.perf_counter()
                metrics=train_generator_lora_epoch(model,tqdm(batches,desc=f'{variant} FIT epoch {epoch+1}'),optimizer,data['fit']['sfr'],
                    gradient_accumulation_steps=settings['gradient_accumulation_steps'],device=device)
                history.append({'epoch':epoch+1,**metrics,'seconds':time.perf_counter()-start})
                atomic_torch(ckpt,{'fingerprint':fingerprint,'variant':variant,'adapter_state':adapter_state(model),'optimizer_state':optimizer.state_dict(),
                    'completed_epochs':epoch+1,'history':history,'settings':settings,'identity_checks':identity_checks,
                    'base_checkpoint':str(checkpoint.relative_to(PROJECT_ROOT)),'format':'additive_delta_LoRA_on_original_SARA'})
                print(variant,'epoch',epoch+1,metrics,flush=True)
            if not any(torch.count_nonzero(p).item() for n,p in trainable.items() if n.endswith('w8g_B')):raise ValueError('New LoRA remained untrained')
            # Verify no original parameter was changed or received gradients.
            for name,p in model.named_parameters():
                if not p.requires_grad and p.grad is not None:raise ValueError('Original parameter has a gradient')
            # Original modules gain an '.original' path component inside attention projections.
            mapped={name.replace('.original.','.'):p for name,p in model.named_parameters() if not name.endswith(('w8g_A','w8g_B'))}
            if set(mapped)!=set(original_parameter_versions) or any(mapped[n]._version!=v for n,v in original_parameter_versions.items()):
                raise ValueError('Original SARA parameter changed during W8-G training')
            optimizer=None
            saved=None
            digest=sha256_file(ckpt);pred=run/f'{variant}_predictions.jsonl';rows=[]
            if pred.exists():
                rows=[json.loads(l) for l in pred.read_text().splitlines()]
                if any(r['checkpoint_sha256']!=digest for r in rows):raise ValueError('Predictions came from different LoRA factors')
            done={(r['question_id'],r['context']) for r in rows}
            expected={(q,c) for q in records for c in ['k4','frozen_controller']}
            if len(done)!=len(rows) or not done<=expected:raise ValueError('Duplicate/extraneous evaluation rows')
            with pred.open('a') as f:
                for q in tqdm(sorted(records),desc=f'{variant} measured validation'):
                    for context,action in [('k4',None),('frozen_controller',decisions.loc[q])]:
                        if (q,context) in done:continue
                        row=generate_row(model,tokenizer,records[q],val['routing'][q],val['sfr'][q],action)
                        row.update(variant=variant,checkpoint_sha256=digest)
                        f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();rows.append(row)
            results[variant]=summarize(rows,baseline);histories[variant]=history
            proofs[variant]={'fresh_SARA_initialization':True,'zero_residual_identity_checks':identity_checks,
                'original_parameter_versions_unchanged':True,'only_new_LoRA_trainable':True,'trainable_params':sum(p.numel() for p in trainable.values()),
                'case_counts':type_counts,'raw_FIT_weight_mean':weight_mean}
            for p in [ckpt,pred,run/f'{variant}_fit_examples.json']:artifacts[p.name]=sha256_file(p)
        finally:
            # trainable/mapped keep references to CUDA parameters; release them before next load.
            trainable=mapped=optimizer=saved=None
            model=None
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    table=[]
    for variant,contexts in results.items():
        for context,metrics in contexts.items():table.append({'variant':variant,'context':context,**metrics})
    pd.DataFrame(table).to_csv(run/'comparison.csv',index=False);artifacts['comparison.csv']=sha256_file(run/'comparison.csv')
    summary={'fingerprint':fingerprint,'smoke_only':smoke,'settings':settings,'results':results,'history':histories,'training_proofs':proofs,
        'artifact_sha256':artifacts,'no_validation_selection':True,'TEST_used':False,'frozen_policy_unchanged':True,
        'evaluation_note':'Original policy and original draft are frozen. STOP returns original draft; only promoted revision uses adapted SARA. k4 is a separate adapted direct-generation experiment.',
        'latency_note':'New pass timings and saved original-draft timings are components, not measured end-to-end deployment latency. Original baseline timing is cached historical data.'}
    atomic_json(run/'summary.json',summary)
    print(pd.DataFrame(table).to_string(index=False),flush=True)
    return summary


def run_ow_lora_training(*args,**kwargs):
    """The old in-memory gamma search is intentionally unavailable."""
    raise RuntimeError('Use run_w8g / run_w8g_lora_training.py: independent FIT-only variants, fixed gamma, real non-oracle generation. Legacy validation-selected gamma search is disabled.')
