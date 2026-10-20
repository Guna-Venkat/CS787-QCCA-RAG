"""W8-J: frozen single-promotion inference on previously inspected DEV.

No training or DEV model selection; all original/G adapters and G0/G1 reported.
"""
from __future__ import annotations
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from src.week8.data import PROJECT_ROOT,load_config,load_week8_split
from src.week8.qcla_experiment import atomic_json,seed_all
from src.week8.generator_features import canonical_k4_context,sha256_file
from src.week8.generation import FrozenRuntimeController,generate_continuation,promoted_prompt
from src.week8.outcome_weighted_training import frozen_policy
from src.week8.hard_case_lora import setup_lora_generator,load_adapter_state
from src.week8.full_system import load_g_variants,read_cache,evaluate_row,summarize,sources as h_sources
from src.week8.semantic import SemanticScorer
from src.week8.utility_training import _validate_saved_routing
from src.week7.w7e.fidelity_allocator import get_semantic_rank_order
from src.week8.final_reporting import paired_interval,report_systems
from src.week6.evaluation import compute_qa_answer_metrics

OUT=PROJECT_ROOT/'results/week8/w8j_final_dev'
CANONICAL=PROJECT_ROOT/'results/week7/w7b_semantic_budget/response_surface/dev_semantic_fixed_k_results.jsonl'


def load_dev_inputs():
    records,sfr,_=load_week8_split('dev')
    if len(records)!=231 or len({str(r['question_id']) for r in records})!=231:raise ValueError('Expected 231 unique DEV queries')
    papers={str(r['paper_id']) for r in records};qids={str(r['question_id']) for r in records}
    cfg=load_config()
    for key in ['fit_manifest','internal_val_manifest']:
        manifest=pd.read_csv(PROJECT_ROOT/cfg['splits'][key],dtype=str)
        if papers&set(manifest.paper_id) or qids&set(manifest.question_id):raise ValueError('DEV split leakage')
    canonical={}
    for line in CANONICAL.read_text().splitlines():
        r=json.loads(line)
        if int(r['k'])==4:
            q=str(r['question_id'])
            if q in canonical:raise ValueError('Duplicate DEV k4 reference')
            canonical[q]=r
    if set(canonical)!=qids:raise ValueError('Incomplete canonical DEV references')
    for r in records:
        q=str(r['question_id']);x=sfr.get(q)
        if x is None or tuple(x.shape)!=(10,4096) or not torch.isfinite(x).all():raise ValueError('Missing real DEV SFR embeddings')
        if len(r['passages'])<10 or str(canonical[q]['paper_id'])!=str(r['paper_id']):raise ValueError('DEV record identity changed')
    return {'records':records,'sfr':sfr,'canonical':canonical}


def dev_routing(record,canonical,scorer):
    texts=[p['text'] for p in record['passages'][:10]]
    scores=scorer.score_query_passages(record['question'],texts).cpu().numpy()
    order=get_semantic_rank_order(semantic_scores=scores,n_passages=10)
    # Retain recorded k4 identities for duplicate/near-tied passages, exactly as W8-C/H do.
    # Fresh scores must still validate this allocation within the existing 1e-5 tolerance.
    full=list(canonical['selected_full_indices'])
    if len(full)!=4 or len(set(full))!=4 or not set(full)<=set(range(10)):raise ValueError('Invalid canonical DEV full allocation')
    order=full+[idx for idx in order if idx not in full]
    rows=[]
    for rank,idx in enumerate(order[4:],5):
        p=record['passages'][idx]
        rows.append({'question_id':str(record['question_id']),'paper_id':str(record['paper_id']),
            'question':record['question'],'semantic_rank':rank,'candidate_passage_index':idx,
            'candidate_text':texts[idx],'bm25_rank':idx+1,'bm25_score':float(p.get('score',0.)),
            'semantic_score':float(scores[idx]),'passage_length':len(texts[idx].split()),
            'draft_answer':str(canonical['prediction'])})
    group=pd.DataFrame(rows)
    _validate_saved_routing(group,texts,scores)
    return group


def smoke_query_ids(records):
    # Fixed identity-only sample; no answer quality or DEV outcomes influence selection.
    qids=sorted(records)
    rng=np.random.default_rng(42)
    return sorted(rng.choice(qids,size=min(20,len(qids)),replace=False).tolist())


def sources(settings,policy_path,adapters):
    _,meta=h_sources(settings,policy_path,adapters)
    cfg=load_config()
    for name in ['src/week8/final_dev.py','src/week8/final_reporting.py','scripts/week8/run_w8j_final_dev_benchmarks.py',
                 cfg['splits']['dev_retrieval'],cfg['splits']['dev_sfr_embeddings'],str(CANONICAL.relative_to(PROJECT_ROOT))]:
        meta['source_sha256'][name]=sha256_file(PROJECT_ROOT/name)
    meta.update(schema='w8j_real_dev_v1',validation_status='DEV previously inspected; no pristine test claim; no selection or tuning; TEST unused')
    return hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),meta


def build_dev_report(run,part,wanted,rows):
    systems={}
    for r in rows:systems.setdefault(r['variant']+'/'+r['revision_mode'],[]).append(r)
    records={str(r['question_id']):r for r in part['records']}
    for line in CANONICAL.read_text().splitlines():
        r=json.loads(line);q=str(r['question_id']);k=int(r['k'])
        if q not in wanted or k not in (2,4,5,6,8):continue
        record=records[q]
        metrics=compute_qa_answer_metrics(r['prediction'],record['gold_answers'],record.get('gold_answers_reformatted',record['gold_answers']))
        # Count exact serialized prompt with checkpoint tokenizer, do not infer tokens from old table.
        from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
        from transformers import AutoTokenizer
        if not hasattr(build_dev_report,'tokenizer'):
            build_dev_report.tokenizer=AutoTokenizer.from_pretrained(str(PROJECT_ROOT/load_config()['sara']['checkpoint_path']),local_files_only=True)
        prompt,_,_,_=build_mixed_fidelity_sara_prompt(str(record['question']).strip(),record['passages'][:10],r['selected_full_indices'],[],r['compressed_indices'],{},build_dev_report.tokenizer,hybrid_medium=False)
        tokens=len(build_dev_report.tokenizer(prompt)['input_ids'])
        systems.setdefault(f'semantic_k{k}',[]).append({'question_id':q,'paper_id':str(record['paper_id']),
            'token_f1':metrics['token_f1'],'final_prompt_tokens':tokens,'cumulative_prompt_tokens':tokens})
    frames={k:pd.DataFrame(v) for k,v in systems.items()}
    for f in frames.values():
        if set(f.question_id)!=wanted or f.question_id.duplicated().any():raise ValueError('Incomplete/duplicate DEV benchmark')
    report_systems(frames,run,'DEV previously inspected; frozen configurations, no tuning; TEST unused')


def run_w8j(output_dir=OUT,*,smoke=False,include_w8g=True,device='cuda',check=False):
    import fcntl
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another W8-J worker owns this directory') from exc
        return _run(folder,smoke,include_w8g,device,check)


def _run(folder,smoke,include_w8g,device,check):
    torch.set_num_threads(4);seed_all(42)
    if not check and str(device).startswith('cuda') and not torch.cuda.is_available() and not (folder/('smoke' if smoke else 'full')/'summary.json').exists():
        raise RuntimeError('CUDA is unavailable in this session. Use a GPU-enabled sara_env kernel/host for the real smoke and full run. --check and CPU inference verification do not require a GPU.')
    policy,_,policy_path=frozen_policy()
    adapters=load_g_variants(include_w8g,policy_path)
    settings={'smoke':smoke,'include_w8g':include_w8g,'device':device,'seed':42,'policy':policy,
        'max_promotions':1,'modes':['G0','G1'],'split':'dev','variants':['original_SARA',*adapters]}
    fingerprint,meta=sources(settings,policy_path,adapters)
    run=folder/('smoke' if smoke else 'full')
    if not check and (run/'summary.json').exists():
        summary=json.loads((run/'summary.json').read_text())
        if summary['fingerprint']!=fingerprint:raise ValueError('Completed W8-J configuration/sources changed; use a new experiment directory')
        for name,digest in summary['artifact_sha256'].items():
            if sha256_file(run/name)!=digest:raise ValueError(f'W8-J artifact changed: {name}')
        print('Verified completed measured W8-J run:',run,flush=True);return summary
    part=load_dev_inputs();records={str(r['question_id']):r for r in part['records']}
    wanted=set(smoke_query_ids(records) if smoke else records)
    if check:
        scorer=SemanticScorer(device='cpu')
        for q in sorted(wanted):dev_routing(records[q],part['canonical'][q],scorer)
        print('Inputs verified:',len(wanted),'queries; live B0-GW + S3/U4-A; paired G0/G1; variants',settings['variants'],'| CUDA available:',torch.cuda.is_available(),'| real GPU smoke required before full execution',flush=True)
        return settings
    run.mkdir(parents=True,exist_ok=True)
    provenance=run/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())['fingerprint']!=fingerprint:raise ValueError('Resume sources changed')
    atomic_json(provenance,{'fingerprint':fingerprint,**meta})
    atomic_json(run/'protocol_frozen_before_evaluation.json',{'fingerprint':fingerprint,**settings,
        'queries':len(wanted),'no_validation_selection':True,'DEV_previously_inspected':True,'QCLA_stacked':False,'DAFiR2_evaluated':False})
    import src
    sara_src=str(PROJECT_ROOT/'Baselines/SARA-main/src')
    if sara_src not in src.__path__:src.__path__.append(sara_src)
    from src.model.loader import load_sara_for_eval
    from transformers import AutoTokenizer
    cfg=load_config();checkpoint=PROJECT_ROOT/cfg['sara']['checkpoint_path']
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True)
    draft_path=run/'live_drafts_and_decisions.jsonl'
    bases=read_cache(draft_path,fingerprint,wanted,lambda r:r['question_id'])
    paths={v:run/f'{v}_predictions.jsonl' for v in settings['variants']}
    expected={(q,m) for q in wanted for m in settings['modes']}
    cache={v:read_cache(p,fingerprint,expected,lambda r:(r['question_id'],r['revision_mode'])) for v,p in paths.items()}
    controller=None
    routing_scorer=SemanticScorer(device='cpu')
    for variant in settings['variants']:
        needs_base=variant=='original_SARA' and set(bases)!=wanted
        needs_revision=set(cache[variant])!=expected
        if not needs_base and not needs_revision:continue
        if str(device).startswith('cuda') and torch.cuda.mem_get_info(device)[0]<18*1024**3:raise RuntimeError('Need at least 18 GiB free GPU memory')
        model,loaded_tokenizer=load_sara_for_eval(str(checkpoint),device=device)
        state=None
        try:
            if loaded_tokenizer('test')['input_ids']!=tokenizer('test')['input_ids']:raise ValueError('Tokenizer differs from checkpoint')
            if variant!='original_SARA':
                state=torch.load(adapters[variant],map_location='cpu',weights_only=False)
                model=setup_lora_generator(model,r=state['settings']['r'],lora_alpha=state['settings']['alpha'],lora_dropout=state['settings']['dropout'])
                load_adapter_state(model,state['adapter_state']);model.eval();state=None
            else:
                model.eval()
                if needs_base:
                    controller=FrozenRuntimeController(json.loads((PROJECT_ROOT/'results/week8/w8e_controller/frozen_controller.json').read_text()))
                    _=controller.scorer.encoder  # model setup is outside per-query timings
            with draft_path.open('a') as draft_file, paths[variant].open('a') as out_file:
                for q in tqdm(sorted(wanted),desc=f'{variant} live G0/G1'):
                    record=records[q];embeddings=part['sfr'].get(q)
                    group=dev_routing(record,part['canonical'][q],routing_scorer)
                    _,_,tokens,_,_,_=canonical_k4_context(group,record,embeddings,tokenizer)
                    group['base_context_tokens']=tokens
                    if embeddings is None:raise ValueError('Missing SFR; no zero fallback')
                    if q not in bases:
                        if variant!='original_SARA':raise ValueError('Shared original-SARA draft is incomplete')
                        start=time.perf_counter()
                        _,prompt,_,compressed,comp,full=canonical_k4_context(group,record,embeddings,tokenizer)
                        routing_seconds=time.perf_counter()-start
                        draft=generate_continuation(model,tokenizer,prompt,comp,device,diagnostics=True)
                        if draft['prediction']!=str(group.iloc[0].draft_answer).strip():raise ValueError(f'{q}: regenerated original k4 differs from canonical; stop instead of changing frozen outcomes')
                        action,inference=controller.decide(record,group,draft['prediction'],draft['diagnostics'])
                        agreement=True  # no stored DEV controller actions are used
                        base={'fingerprint':fingerprint,'question_id':q,'paper_id':str(record['paper_id']),'draft':draft,
                            'decision':action,'inference':inference,'full_indices':full,'compressed_indices':compressed,
                            'base_pipeline_seconds':routing_seconds+draft['generation_seconds']+draft['diagnostics_seconds']+inference['decision_seconds'],
                            'reference_action_match':bool(agreement)}
                        draft_file.write(json.dumps(base,allow_nan=False)+'\n');draft_file.flush();bases[q]=base
                    base=bases[q]
                    for mode in settings['modes']:
                        if (q,mode) in cache[variant]:continue
                        revision=None
                        if base['decision']['selected_action']=='PROMOTE':
                            start=time.perf_counter()
                            prompt,comp,full,compressed=promoted_prompt(record,group,embeddings,tokenizer,base['decision'],mode,base['draft']['prediction'])
                            prompt_seconds=time.perf_counter()-start
                            revision=generate_continuation(model,tokenizer,prompt,comp,device)
                            revision.update(prompt_build_seconds=prompt_seconds,full_indices=full,compressed_indices=compressed)
                        row=evaluate_row(record,base,revision,variant,mode,fingerprint)
                        row['reference_action_match']=None
                        out_file.write(json.dumps(row,allow_nan=False)+'\n');out_file.flush();cache[variant][(q,mode)]=row
        finally:
            state=None;model=None;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    rows=[r for variant in settings['variants'] for r in cache[variant].values()]
    if len(rows)!=len(expected)*len(settings['variants']):raise ValueError('Incomplete DEV evaluation')
    if smoke and {b['decision']['selected_action'] for b in bases.values()}!={'STOP','PROMOTE'}:
        raise RuntimeError('Smoke has not exercised both STOP and PROMOTE. Inspect before full evaluation; no branch override is allowed.')
    table=summarize(rows);
    for result in table:result['reference_action_agreement']=None
    pd.DataFrame(table).to_csv(run/'comparison.csv',index=False)
    artifacts={p.name:sha256_file(p) for p in [draft_path,*paths.values(),run/'comparison.csv']}
    summary={'fingerprint':fingerprint,'settings':settings,'smoke_only':smoke,'results':table,'artifact_sha256':artifacts,
        'live_gate_and_selector_inference':True,'no_validation_selection':True,'TEST_used':False,'QCLA_stacked':False,'DAFiR2_evaluated':False,
        'reference_action_agreement':None,'DEV_previously_inspected':True,'split':'dev',
        'latency_note':meta['latency_scope'],'frozen_sources_unchanged':True}
    build_dev_report(run,part,wanted,rows)
    for name in ['benchmark.csv','paired_paper_intervals.csv','report_notes.json','quality_cost.png']:
        summary['artifact_sha256'][name]=sha256_file(run/name)
    current,_=sources(settings,policy_path,adapters)
    if current!=fingerprint:raise ValueError('Sources changed during DEV evaluation; no completed summary written')
    summary['smoke_branch_coverage']=sorted({b['decision']['selected_action'] for b in bases.values()}) if smoke else None
    atomic_json(run/'summary.json',summary)
    print(pd.DataFrame(table).to_string(index=False),flush=True)
    return summary
