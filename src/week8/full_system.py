"""W8-H real inference and matched G0/G1 single-promotion evaluation.

No training, policy tuning or validation-selected model stacking. The original
SARA draft and live frozen-policy decision are shared by all revision variants.
Retrieval and SFR vectors are precomputed benchmark inputs, not measured online
retrieval. Latency fields report timed query-compute components, not deployment
end-to-end wall time or cold model loading.
"""
from __future__ import annotations
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from src.week8.data import PROJECT_ROOT,load_config
from src.week8.qcla_experiment import load_inputs,atomic_json,seed_all
from src.week8.generator_features import canonical_k4_context,sha256_file
from src.week8.generation import FrozenRuntimeController,generate_continuation,promoted_prompt
from src.week8.outcome_weighted_training import frozen_policy,sources as g_sources
from src.week8.hard_case_lora import setup_lora_generator,load_adapter_state
from src.week6.evaluation import compute_qa_answer_metrics

OUT=PROJECT_ROOT/'results/week8/w8h_full_system'


def load_g_variants(include,policy_path):
    if not include:return {}
    folder=PROJECT_ROOT/'results/week8/w8g_lora/full';summary=json.loads((folder/'summary.json').read_text())
    fp,_=g_sources(summary['settings'],policy_path)
    if fp!=summary['fingerprint'] or summary['smoke_only'] or summary['TEST_used']:raise ValueError('Completed real W8-G checkpoint provenance changed')
    for name,digest in summary['artifact_sha256'].items():
        if sha256_file(folder/name)!=digest:raise ValueError(f'W8-G artifact integrity failed: {name}')
    return {name:folder/f'{name}.pt' for name in ['uniform_SFT','HC_LoRA','OW_LoRA']}


def sources(settings,policy_path,adapters):
    _,base=g_sources(settings,policy_path)
    hashes=base['source_sha256']
    names=['src/week8/generation.py','src/week8/full_system.py','scripts/week8/run_w8h_full_evaluation.py',
        'src/week8/gate_diagnostics.py','src/week8/selective_controller.py','src/week8/semantic.py','src/week8/marginal_information.py',
        'src/week8/utility_models.py','src/week8/utility_dataset.py',
        'results/week8/w8e_controller/b0_gw_gate.joblib','results/week8/w8c_utility/selected_neural.pt',
        'results/week8/w8c_utility/w8c_frozen_selector_config.json']
    for name in names:hashes[name]=sha256_file(PROJECT_ROOT/name)
    for path in adapters.values():hashes[str(path.relative_to(PROJECT_ROOT))]=sha256_file(path)
    if adapters:hashes['results/week8/w8g_lora/full/summary.json']=sha256_file(PROJECT_ROOT/'results/week8/w8g_lora/full/summary.json')
    meta={'schema':'w8h_live_frozen_controller_v1','settings':settings,'source_sha256':hashes,'environment':base['environment'],
        'inference':'New original-SARA draft, real generated-token diagnostics, frozen gain-weighted gate, new MiniLM/U4-A inference; fixed FIT-selected thresholds.',
        'matched_comparison':'G0/G1 share the exact same current draft, action and promoted passage. G1 appends only draft/instructions inside the user chat turn.',
        'variants':'Original SARA and optionally ALL three predeclared W8-G revision adapters. No validation-winner selection.',
        'excluded':'DAFiR-2 lacks frozen second-state training/calibration. QCLA changes base-draft distribution and is not automatically stacked with the original gate.',
        'validation_status':'Previously inspected INTERNAL-VAL development confirmation; TEST unused.',
        'latency_scope':'Timed draft generation, diagnostics, CPU gate/selector and revision generation/prompt components. Cached retrieval/SFR and cold model loading excluded. Branch component sums are not deployment end-to-end wall times.'}
    return hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),meta


def smoke_ids(part,reference):
    qids=[]
    for action in ['STOP','PROMOTE']:
        options=sorted(q for q in part['routing'] if reference.loc[q,'selected_action']==action)
        if not options:raise ValueError(f'Missing real {action} smoke case')
        qids.append(options[0])
    for r in sorted(part['records'],key=lambda r:str(r['question_id'])):
        q=str(r['question_id'])
        if q not in qids and str(r['question'])!=str(r['question']).strip():qids.append(q);break
    return sorted(set(qids))


def read_cache(path,fingerprint,wanted,key):
    if not path.exists():return {}
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    result={key(r):r for r in rows}
    if len(result)!=len(rows) or not set(result)<=wanted:raise ValueError('Duplicate or extraneous W8-H cache rows')
    if any(r['fingerprint']!=fingerprint for r in rows):raise ValueError('Stale W8-H measurement cache')
    return result


def evaluate_row(record,base,revision,variant,mode,fingerprint):
    decision=base['decision'];promote=decision['selected_action']=='PROMOTE'
    prediction=revision['prediction'] if promote else base['draft']['prediction']
    # Outcome scoring occurs only after the live action and generated answer.
    gold=record['gold_answers'];formatted=record.get('gold_answers_reformatted',gold)
    metrics=compute_qa_answer_metrics(prediction,gold,formatted)
    base_metrics=compute_qa_answer_metrics(base['draft']['prediction'],gold,formatted)
    base_cost=base['draft']['prompt_tokens']
    final_cost=revision['prompt_tokens'] if promote else base_cost
    revision_seconds=revision['generation_seconds'] if promote else 0.
    prompt_seconds=revision['prompt_build_seconds'] if promote else 0.
    compute_seconds=base['base_pipeline_seconds']+revision_seconds+prompt_seconds
    return {'fingerprint':fingerprint,'question_id':base['question_id'],'paper_id':base['paper_id'],
        'variant':variant,'revision_mode':mode,'selected_action':decision['selected_action'],
        'candidate_passage_index':decision['candidate_passage_index'],'semantic_rank':decision['semantic_rank'],'bm25_rank':decision['bm25_rank'],
        'proposed_candidate_passage_index':decision['proposed_candidate_passage_index'],
        'gate_probability':base['inference']['gate_probability'],'prediction':prediction,'draft_answer':base['draft']['prediction'],
        'token_f1':metrics['token_f1'],'exact_match':metrics['exact_match'],'base_f1':base_metrics['token_f1'],
        'delta_f1':metrics['token_f1']-base_metrics['token_f1'],
        'num_generation_passes':2 if promote else 1,'final_prompt_tokens':final_cost,
        'cumulative_prompt_tokens':base_cost+(final_cost if promote else 0),
        'generated_tokens':base['draft']['generated_tokens']+(revision['generated_tokens'] if promote else 0),
        'draft_generation_seconds':base['draft']['generation_seconds'],'diagnostics_seconds':base['draft']['diagnostics_seconds'],
        'decision_seconds':base['inference']['decision_seconds'],'revision_generation_seconds':revision_seconds,
        'timed_query_compute_seconds':compute_seconds,'end_to_end_latency_sec':None,
        'reference_action_match':base['reference_action_match'],'full_indices':revision['full_indices'] if promote else base['full_indices'],
        'compressed_indices':revision['compressed_indices'] if promote else base['compressed_indices']}


def summarize(rows):
    f=pd.DataFrame(rows);result=[]
    for (variant,mode),g in f.groupby(['variant','revision_mode'],sort=False):
        p=g.selected_action.eq('PROMOTE');n=int(p.sum());d=g.delta_f1
        result.append({'variant':variant,'revision_mode':mode,'queries':len(g),'downstream_f1':float(g.token_f1.mean()),
            'base_k4_f1':float(g.base_f1.mean()),'gain_over_k4':float(d.mean()),'exact_match':float(g.exact_match.mean()),
            'promotion_rate':float(p.mean()),'hurt_rate_all_queries':float((d<-.01).mean()),
            'help_precision_promotions':float((d>.01).sum()/n) if n else 0.,'hurt_rate_promotions':float((d<-.01).sum()/n) if n else 0.,
            'mean_final_prompt_tokens':float(g.final_prompt_tokens.mean()),'mean_cumulative_prompt_tokens':float(g.cumulative_prompt_tokens.mean()),
            'mean_generated_tokens':float(g.generated_tokens.mean()),'mean_timed_query_compute_seconds':float(g.timed_query_compute_seconds.mean()),
            'reference_action_agreement':float(g.reference_action_match.mean()),'end_to_end_latency_sec':None})
    return result


def run_w8h(output_dir=OUT,*,smoke=False,include_w8g=False,device='cuda',check=False):
    import fcntl
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another W8-H worker owns this output directory') from exc
        return _run(folder,smoke,include_w8g,device,check)


def _run(folder,smoke,include_w8g,device,check):
    torch.set_num_threads(4);seed_all(42)
    if not check and str(device).startswith('cuda') and not torch.cuda.is_available() and not (folder/('smoke' if smoke else 'full')/'summary.json').exists():
        raise RuntimeError('CUDA is unavailable in this session. Use a GPU-enabled sara_env kernel/host for the real smoke and full run. --check and CPU inference verification do not require a GPU.')
    policy,reference,policy_path=frozen_policy()
    adapters=load_g_variants(include_w8g,policy_path)
    settings={'smoke':smoke,'include_w8g':include_w8g,'device':device,'seed':42,'policy':policy,
        'max_promotions':1,'modes':['G0','G1'],'variants':['original_SARA',*adapters]}
    fingerprint,meta=sources(settings,policy_path,adapters)
    run=folder/('smoke' if smoke else 'full')
    if not check and (run/'summary.json').exists():
        summary=json.loads((run/'summary.json').read_text())
        if summary['fingerprint']!=fingerprint:raise ValueError('Completed W8-H configuration/sources changed; use a new experiment directory')
        for name,digest in summary['artifact_sha256'].items():
            if sha256_file(run/name)!=digest:raise ValueError(f'W8-H artifact changed: {name}')
        print('Verified completed measured W8-H run:',run,flush=True);return summary
    part=load_inputs()['internal_val'];records={str(r['question_id']):r for r in part['records']}
    if set(records)!=set(reference.index):raise ValueError('Frozen policy membership differs from validation')
    wanted=set(smoke_ids(part,reference) if smoke else records)
    if check:
        print('Inputs verified:',len(wanted),'queries; live B0-GW + S3/U4-A; paired G0/G1; variants',settings['variants'],'| CUDA available:',torch.cuda.is_available(),'| real GPU smoke required before full execution',flush=True)
        return settings
    run.mkdir(parents=True,exist_ok=True)
    provenance=run/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())['fingerprint']!=fingerprint:raise ValueError('Resume sources changed')
    atomic_json(provenance,{'fingerprint':fingerprint,**meta})
    atomic_json(run/'protocol_frozen_before_evaluation.json',{'fingerprint':fingerprint,**settings,
        'queries':len(wanted),'no_validation_selection':True,'QCLA_stacked':False,'DAFiR2_evaluated':False})
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
                    record=records[q];group=part['routing'][q];embeddings=part['sfr'].get(q)
                    if embeddings is None:raise ValueError('Missing SFR; no zero fallback')
                    if q not in bases:
                        if variant!='original_SARA':raise ValueError('Shared original-SARA draft is incomplete')
                        start=time.perf_counter()
                        _,prompt,_,compressed,comp,full=canonical_k4_context(group,record,embeddings,tokenizer)
                        routing_seconds=time.perf_counter()-start
                        draft=generate_continuation(model,tokenizer,prompt,comp,device,diagnostics=True)
                        if draft['prediction']!=str(group.iloc[0].draft_answer).strip():raise ValueError(f'{q}: regenerated original k4 differs from canonical; stop instead of changing frozen outcomes')
                        action,inference=controller.decide(record,group,draft['prediction'],draft['diagnostics'])
                        ref=reference.loc[q]
                        agreement=action['selected_action']==ref.selected_action
                        if agreement and action['selected_action']=='PROMOTE':agreement=int(action['candidate_passage_index'])==int(ref.candidate_passage_index)
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
                            if variant=='original_SARA' and mode=='G0':
                                ref=group[group.candidate_passage_index.astype(int)==int(base['decision']['candidate_passage_index'])].iloc[0]
                                if revision['prediction']!=str(ref.promoted_prediction).strip():raise ValueError('Original G0 differs from canonical promotion; inspect prompt/model parity')
                        row=evaluate_row(record,base,revision,variant,mode,fingerprint)
                        out_file.write(json.dumps(row,allow_nan=False)+'\n');out_file.flush();cache[variant][(q,mode)]=row
        finally:
            state=None;model=None;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    rows=[r for variant in settings['variants'] for r in cache[variant].values()]
    table=summarize(rows);pd.DataFrame(table).to_csv(run/'comparison.csv',index=False)
    artifacts={p.name:sha256_file(p) for p in [draft_path,*paths.values(),run/'comparison.csv']}
    summary={'fingerprint':fingerprint,'settings':settings,'smoke_only':smoke,'results':table,'artifact_sha256':artifacts,
        'live_gate_and_selector_inference':True,'no_validation_selection':True,'TEST_used':False,'QCLA_stacked':False,'DAFiR2_evaluated':False,
        'reference_action_agreement':float(np.mean([r['reference_action_match'] for r in bases.values()])),
        'latency_note':meta['latency_scope'],'frozen_sources_unchanged':True}
    atomic_json(run/'summary.json',summary)
    print(pd.DataFrame(table).to_string(index=False),flush=True)
    return summary
