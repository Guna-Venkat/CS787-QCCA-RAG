"""Audited saved baseline adapters. Missing and summary-only rows stay explicit."""
from __future__ import annotations
import json,re
from pathlib import Path
import pandas as pd
from .provenance import ROOT,file_hash,digest,branch_guard
from .caching import read_jsonl,atomic_json
from .evaluation import annotate_outcome,policy_summary,paired_paper_bootstrap

EVAL_HASH=lambda:file_hash(ROOT/'src/week6/evaluation.py')
NOTEBOOK=ROOT/'Baselines/SARA-main/notebooks/sara_baseline_checkpoints.ipynb'
FAMILIES=['ORIGINAL_SARA','BM25_FIXED','SEMANTIC_FIXED','W8_INTERVENTION','HC_LORA','W8_SECONDARY']


def aligned_rows(left,right):
    for rows in (left,right):
        keys=[(str(r['question_id']),str(r['paper_id'])) for r in rows]
        if len(keys)!=len(set(keys)) or len({q for q,p in keys})!=len(keys):raise ValueError('Duplicate baseline question/paper IDs')
    keys=lambda rows:{(str(r['question_id']),str(r['paper_id'])) for r in rows}
    if not left or keys(left)!=keys(right):raise ValueError('Baseline query/paper IDs differ')
    for field in ('split','evaluation_hash'):
        if len({r.get(field) for r in left+right})!=1 or left[0].get(field) is None:raise ValueError('Baseline split/scorer mismatch')


def entry(baseline_id,family,name,week,path,split,*,rows=None,status='AVAILABLE',note='',secondary=False):
    path=Path(path);rows=rows or []
    return dict(baseline_id=baseline_id,family=family,display_name=name,source_week=week,source_artifact=str(path),
        split=split,query_ids=[r['question_id'] for r in rows] if rows else None,
        paper_ids=[r['paper_id'] for r in rows] if rows else None,num_queries=len(rows) if rows else None,
        generator='canonical Mistral/SARA' if family!='W8_SECONDARY' else name,retriever='BM25' if family=='BM25_FIXED' else 'all-MiniLM-L6-v2 semantic' if family in ('SEMANTIC_FIXED','W8_INTERVENTION','HC_LORA') else 'see source',
        fidelity_policy=name,revision_mode='G0' if name.endswith('_G0') else 'G1' if name.endswith('_G1') else None,
        prompt_tokens=mean(rows,'final_prompt_tokens'),cumulative_tokens=mean(rows,'total_query_prompt_tokens'),
        generated_tokens=mean(rows,'total_query_generated_tokens'),latency=mean(rows,'revision_latency'),
        F1=mean(rows,'f1'),EM=mean(rows,'exact_match'),artifact_hash=file_hash(path) if path.is_file() else None,
        evaluation_hash=EVAL_HASH() if rows else None,status=status,note=note,secondary=secondary,
        paired_bootstrap_available=bool(rows),rows=rows)


def saved_directory(relative):
    current=ROOT/relative
    if (current/'summary.json').exists():return current
    manifest_path=ROOT/'results/week9/baselines/w8_import_manifest.json'
    if manifest_path.exists():
        imported=json.loads(manifest_path.read_text())
        key=relative+'/summary.json'
        if key in imported['files']:
            parent=ROOT/Path(imported['files'][key]['imported_path']).parent
            for name,value in imported['files'].items():
                if str(Path(name).parent)==relative and file_hash(ROOT/value['imported_path'])!=value['sha256']:raise ValueError('Imported W8 artifact changed')
            return parent
    return current


def mean(rows,key):
    vals=[r[key] for r in rows if r.get(key) is not None]
    return float(sum(vals)/len(vals)) if vals and len(vals)==len(rows) else None


def normalize_saved(saved,records,split,policy,*,mode=None,kind='standard'):
    byid={r['inference']['question_id']:r for r in records};out=[];seen=set()
    for row in saved:
        q=str(row['question_id'])
        if q not in byid or (mode is not None and row.get('revision_mode')!=mode):continue
        if kind=='k4' and row.get('evaluation_context',row.get('mode',row.get('context'))) not in (None,'k4'):continue
        if q in seen:raise ValueError('Duplicate baseline outcomes')
        seen.add(q);rec=byid[q];pid=rec['inference']['paper_id']
        if str(row['paper_id'])!=pid:raise ValueError('Saved baseline paper identity mismatch')
        prompt=row.get('final_prompt_tokens',row.get('prompt_tokens',row.get('input_tokens',row.get('prompt_token_count'))))
        cumulative=row.get('cumulative_prompt_tokens',prompt)
        result=dict(question_id=q,paper_id=pid,policy=policy,prediction=row['prediction'],split=split,
            final_prompt_tokens=prompt,total_query_prompt_tokens=cumulative,
            total_query_generated_tokens=row.get('generated_tokens',row.get('generated_token_count')),
            num_lm_calls=row.get('num_generation_passes',row.get('generation_passes',1)),
            revision_latency=row.get('generation_latency_sec',row.get('generation_latency',row.get('latency',row.get('new_generation_latency_sec')))),
            evaluation_hash=EVAL_HASH(),evidence_text='',faithfulness=None,
            promotion_rate=float(row.get('selected_action')=='PROMOTE'),extraction_rate=0.,
            measured_end_to_end_latency=None)
        out.append(annotate_outcome(result,rec['inference'],rec['evaluation']))
    if out and set(r['question_id'] for r in out)!=set(byid):raise ValueError('Incomplete baseline membership; do not pair partial outcomes')
    return out


def build_registry(cfg,split,records,tokenizer=None,*,write=True):
    from .data import resolve
    from .backend import load_tokenizer
    if split not in ('fit','fit_heldout','internal_val','dev'):raise PermissionError('Baseline audit does not open TEST')
    if tokenizer is None:tokenizer=load_tokenizer(cfg)
    result=[]
    # Programmatically inventory the sole canonical checkpoint notebook; no invented benchmark metrics.
    nb=json.loads(NOTEBOOK.read_text());text='\n'.join(''.join(c['source']) for c in nb['cells'])
    for label,needle in [('Standard RAG','rag_cfg'),('SARA','sara_cfg')]:
        if needle in text:
            e=entry('ORIGINAL_SARA/'+label,'ORIGINAL_SARA',label,1,NOTEBOOK,'checkpoint_audit',status='CHECKPOINT_ONLY',
                note='Canonical notebook profiles this checkpoint; no full query-level QA benchmark is present. No paired metrics inferred.')
            e['generator']='Mistral-7B-Instruct-v0.2 + '+('Standard RAG LoRA, projector-free' if label=='Standard RAG' else 'trained SARA LoRA/projector')
            result.append(e)
    root=ROOT/'results/week4'
    # Include every fixed-policy row in the canonical W4 benchmark and validation summaries.
    for path in sorted(root.rglob('*.csv')):
        if path.name not in ('policy_comparison.csv','policy_pareto_points.csv','qcca_v2_policy_summary.csv','baseline_comparison.csv'):continue
        frame=pd.read_csv(path)
        column=next((c for c in ('system','policy','baseline') if c in frame),None)
        if column is None:continue
        for r in frame.to_dict('records'):
            name=str(r[column])
            if not re.search(r'(static|fixed).*k\s*[=_]?\s*\d',name,re.I):continue
            e=entry('W4/'+str(path.relative_to(root))+'/'+name,'BM25_FIXED',name,4,path,'historical_week4_subset',status='SUMMARY_ONLY',
                note='Exact stored W4 row; membership/scorer not established. Descriptive only; no paired bootstrap.')
            e['F1']=r.get('mean_f1',r.get('f1'));e['prompt_tokens']=r.get('mean_context_tokens');result.append(e)
    # Canonical query-level BM25 and semantic fixed policies. Never regenerate answers.
    paths=[('BM25_FIXED',5 if split!='dev' else 2,ROOT/('results/week5/train_response_surface/train_fixed_k_results.jsonl' if split!='dev' else 'results/week2/raw/dev_fixed_k_matrix.jsonl')),
           ('SEMANTIC_FIXED',7,ROOT/('results/week7/w7b_semantic_budget/response_surface/train_semantic_fixed_k_results.jsonl' if split!='dev' else 'results/week7/w7b_semantic_budget/response_surface/dev_semantic_fixed_k_results.jsonl'))]
    wanted={r['inference']['question_id']:r for r in records}
    from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
    for family,week,path in paths:
        if not path.exists():
            result.append(entry(family,family,family,week,path,split,status='MISSING'));continue
        saved=read_jsonl(path)
        for k in sorted({int(r['k']) for r in saved}):
            rows=[]
            for r in saved:
                if int(r['k'])!=k or str(r['question_id']) not in wanted:continue
                rec=wanted[str(r['question_id'])];inf=rec['inference']
                full=r.get('selected_full_indices',list(range(k)));heavy=r.get('compressed_indices',list(range(k,10)))
                prompt,_,_,_=build_mixed_fidelity_sara_prompt(inf['question'],inf['passages'],full,[],heavy,{},tokenizer)
                n=len(tokenizer.encode(prompt));rows.append(dict(r,prompt_tokens=n,cumulative_prompt_tokens=n))
            name=('semantic' if family=='SEMANTIC_FIXED' else 'BM25')+f'_k{k}'
            normalized=normalize_saved(rows,records,split,name)
            if normalized:result.append(entry(name,family,name,week,path,split,rows=normalized))
    # Mandatory final intervention: final H/J prediction artifacts, NEVER the earlier W8-E replay.
    final=saved_directory('results/week8/w8j_final_dev/full' if split=='dev' else 'results/week8/w8h_full_system/full')
    for variant,family in [('original_SARA','W8_INTERVENTION'),('HC_LoRA','HC_LORA'),('uniform_SFT','W8_SECONDARY'),('OW_LoRA','W8_SECONDARY')]:
        for mode in ('G0','G1'):
            path=final/f'{variant}_predictions.jsonl';name=variant+'_'+mode
            if split in ('fit','fit_heldout') or not path.exists():
                result.append(entry(name,family,name,8,path,split,status='MISSING',secondary=mode=='G1' or family=='W8_SECONDARY',
                    note='Requires real final same-split predictions; full-FIT trained systems cannot be relabelled as FIT OOF.'))
            else:
                summary_path=final/'summary.json'
                if not summary_path.exists():raise ValueError('Final W8 predictions require their canonical summary/provenance')
                summary=json.loads(summary_path.read_text())
                if summary.get('smoke_only') or summary.get('TEST_used'):raise ValueError('Invalid final W8 benchmark provenance')
                expected=summary.get('artifact_sha256',{}).get(path.name)
                if expected!=file_hash(path):raise ValueError('Final W8 prediction checksum mismatch')
                rows=normalize_saved(read_jsonl(path),records,split,name,mode=mode)
                e=entry(name,family,name,8,path,split,rows=rows,secondary=mode=='G1' or family=='W8_SECONDARY')
                e.update(fidelity_policy='Canonical semantic k4; at most one promotion to 5 FULL / 5 HEAVY',promotion_policy=summary['settings'].get('policy'),generation_mode=mode,prompt_version=summary['fingerprint'],
                    frozen_protocol=summary['settings'],source_summary_hash=file_hash(summary_path),
                    generator='Original SARA draft; '+variant+' revision',metrics_note='Saved predictions rescored by the current common scorer; no generation repeated')
                provenance=final/'provenance.json'
                if provenance.exists():
                    value=json.loads(provenance.read_text())
                    e.update(selector_gate=value.get('inference'),source_provenance_hash=file_hash(provenance))
                result.append(e)
    # Direct HC-LoRA k4 is distinct from its G0 intervention.
    path=saved_directory('results/week8/w8g_lora/full')/'HC_LoRA_predictions.jsonl'
    rows=[]
    if split=='internal_val' and path.exists():
        saved=[r for r in read_jsonl(path) if r.get('evaluation_context',r.get('mode',r.get('context')))=='k4']
        rows=normalize_saved(saved,records,split,'HC_LoRA_k4')
    result.append(entry('HC_LoRA_k4','HC_LORA','HC_LoRA_k4',8,path,split,rows=rows,status='AVAILABLE' if rows else 'MISSING',note='Direct adapted k4; not HC-LoRA/G0.'))
    path=ROOT/'results/week8/w8f_qcla/full/qcla_1_internal_val.jsonl'
    if split=='internal_val' and path.exists():result.append(entry('QCLA_1','W8_SECONDARY','QCLA_1',8,path,split,
        rows=normalize_saved(read_jsonl(path),records,split,'QCLA_1'),secondary=True))
    for e in result:
        if not e['rows'] and e['status']=='AVAILABLE':e.update(status='MISSING',note='No same-split query predictions')
    if write:
        branch_guard();folder=resolve(cfg['output_root'])/'baselines'/split
        atomic_json(folder/'baseline_manifest.json',dict(split=split,entries=[{k:v for k,v in e.items() if k!='rows'} for e in result],
            no_historical_scores_retyped=True,original_notebook=str(NOTEBOOK),original_notebook_hash=file_hash(NOTEBOOK)))
        atomic_json(resolve(cfg['output_root'])/'baselines/baseline_manifest.json',dict(latest_split=split,split_manifest=str(folder/'baseline_manifest.json'),
            families=FAMILIES,note='Split-specific manifests preserve separate query sets.'))
    return result


def final_tables(cfg,split,records,method_rows,directory,tokenizer=None):
    entries=build_registry(cfg,split,records,tokenizer)
    main=[];extended=[];unpaired=[];comparisons={}
    method_rows=[dict(r,split=split,evaluation_hash=EVAL_HASH()) for r in method_rows]
    for policy in sorted({r['policy'] for r in method_rows}):
        rows=[r for r in method_rows if r['policy']==policy]
        main.append(dict(baseline_id=policy,family='W9',split=split,**policy_summary(rows)))
    reference=[r for r in method_rows if r['policy']=='B0']
    for e in entries:
        row={k:v for k,v in e.items() if k not in ('rows','query_ids','paper_ids')}
        for old,new in [('F1','f1'),('EM','exact_match'),('prompt_tokens','final_prompt_tokens'),('cumulative_tokens','total_query_prompt_tokens'),('generated_tokens','total_query_generated_tokens'),('num_queries','n_queries')]:
            row[new]=row.pop(old)
        if e['rows']:
            try:aligned_rows(e['rows'],reference)
            except ValueError as exc:
                row['comparison_note']=str(exc);unpaired.append(row);continue
            comparisons[e['baseline_id']]=paired_paper_bootstrap(e['rows'],reference)
            (extended if e['secondary'] else main).append(row)
        else:unpaired.append(row)
    directory=Path(directory)
    pd.DataFrame(main).to_csv(directory/'w9h_main_method_table.csv',index=False)
    pd.DataFrame(extended,columns=None if extended else ['baseline_id','family','split','status','f1','exact_match']).to_csv(directory/'w9h_extended_ablation_table.csv',index=False)
    pd.DataFrame(unpaired).to_csv(directory/'w9h_unpaired_or_missing_baselines.csv',index=False)
    atomic_json(directory/'w9h_registered_baseline_intervals.json',comparisons)
    # Cross-model results remain a separate view with explicit textual interface / precision.
    robustness=[]
    if cfg.get('generator_robustness_exports'):
        from .config import load_frozen
        panel=json.loads(Path(cfg.get('robustness_panel','results/week9/preflight/frozen_model_panel.json')).read_text())
        if panel.get('panel_hash')!=digest({k:v for k,v in panel.items() if k!='panel_hash'}) or not panel.get('frozen') or panel.get('selection_split')!='fit':raise ValueError('Robustness requires the verified FIT model panel')
        method=load_frozen(Path(cfg['output_root'])/'frozen/w9h_frozen_config.json','h')
        if cfg.get('model_panel_hash')!=panel['panel_hash'] or cfg.get('frozen_method_hash')!=method['settings_hash']:raise ValueError('Configured robustness hashes do not match actual frozen artifacts')
    for name in cfg.get('generator_robustness_exports',[]):
        path=Path(name);export=json.loads(path.read_text())
        if (export.get('split')!=split or not export.get('model_panel_hash') or export['model_panel_hash']!=cfg.get('model_panel_hash') or
            not export.get('frozen_method_hash') or export['frozen_method_hash']!=cfg.get('frozen_method_hash') or
            export.get('first_pass_sara_model')!='mistralai/Mistral-7B-Instruct-v0.2'):
            raise ValueError('Robustness export split/panel/frozen-method mismatch')
        allowed={r['model_id']:r for r in panel['model_panel']}
        choice=allowed.get(export['model_id'])
        if choice is None or choice['revision']!=export['model_revision']:raise ValueError('Generator is outside the frozen panel/revision')
        from .model_registry import runtime_identity
        runtime=runtime_identity(choice['model_id'],choice['revision'],choice['precision'])
        if export['quantization']!=runtime['quantization']:raise ValueError('Generator quantization differs from its frozen panel')
        aligned_rows(export['rows'],reference)
        byid={r['inference']['question_id']:r for r in records};verified=[]
        for row in export['rows']:
            record=byid[row['question_id']]
            scored=annotate_outcome(row,record['inference'],record['evaluation'])
            if abs(scored['f1']-row['f1'])>1e-8:raise ValueError('Robustness export disagrees with the current evaluator')
            verified.append(scored)
        calls=[c for r in verified for c in r.get('model_calls',[]) if c.get('model_id')==choice['model_id']]
        allocated=[c['peak_memory_allocated'] for c in calls if c.get('peak_memory_allocated') is not None]
        reserved=[c['peak_memory_reserved'] for c in calls if c.get('peak_memory_reserved') is not None]
        chars=[c['input_characters'] for c in calls if c.get('input_characters') is not None]
        robustness.append(dict(model_id=export['model_id'],interface='textual evidence; canonical first-pass SARA',
            model_revision=export['model_revision'],dtype=runtime['dtype'],quantization=export['quantization'],
            peak_memory_allocated=max(allocated) if allocated else None,peak_memory_reserved=max(reserved) if reserved else None,
            mean_input_characters=sum(chars)/len(chars) if chars else None,
            artifact_hash=file_hash(path),**policy_summary(verified)))
    pd.DataFrame(robustness,columns=None if robustness else ['model_id','interface','status']).to_csv(directory/'w9h_generator_robustness.csv',index=False)
    return entries


def baseline_source_paths():
    paths=[NOTEBOOK,ROOT/'results/week5/train_response_surface/train_fixed_k_results.jsonl',
        ROOT/'results/week2/raw/dev_fixed_k_matrix.jsonl',ROOT/'results/week8/w8f_qcla/full/qcla_1_internal_val.jsonl']
    for folder in ['results/week8/w8h_full_system/full','results/week8/w8j_final_dev/full','results/week8/w8g_lora/full']:
        paths.extend(saved_directory(folder).glob('*predictions.jsonl'))
        paths.extend(saved_directory(folder).glob('summary.json'))
    for name in ('policy_comparison.csv','policy_pareto_points.csv','qcca_v2_policy_summary.csv','baseline_comparison.csv'):
        paths.extend((ROOT/'results/week4').rglob(name))
    paths.append(ROOT/'results/week9/baselines/w8_import_manifest.json')
    return sorted(set(p for p in paths if p.is_file()))
