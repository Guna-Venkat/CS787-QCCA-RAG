"""Reviewed fallback freeze; original C/D/attribution scientific hashes stay unchanged.

Restricts confirmation to existing MAIN controls with complete matched VAL at both
budgets. A qualified but unevaluated compressor is never auto-enrolled.
"""
import json,subprocess
from collections import defaultdict
import pandas as pd
from src.week12.protocol import ROOT,root,identity,sha,digest,immutable,utc,cohort,guard

def load_saved_review(cfg):
    guard();p=root(cfg)/'analysis';status=json.loads((p/'status.json').read_text())
    if status.get('synthetic') or not status.get('saved_result_analysis_only'):raise ValueError('Complete real H saved analysis required')
    for phase in ['C','D']:
        s=json.loads((root(cfg)/'validation'/phase/'status.json').read_text())
        meta=json.loads((root(cfg)/'validation'/phase/'rows/identity.json').read_text())
        if not s['complete'] or not s['full_generation'] or meta['code_hash']!=identity(cfg)['code_hash'] or meta['config_hash']!=digest(cfg):raise ValueError('Incomplete or incompatible C/D evidence')
    def read(path):
        try:return pd.read_csv(path)
        except pd.errors.EmptyDataError:return pd.DataFrame()
    ext=root(cfg)/'c_extension/v1/analysis'
    result=dict(summary=read(p/'summary.csv'),comparisons=read(p/'primary_comparisons.csv'),fusion_comparisons=read(p/'fusion_diagnostic_comparisons.csv'),specialist_references=read(p/'specialist_references.csv'),figures=json.loads((p/'figures.json').read_text()),c_extension=dict(summary=read(ext/'merged_validation_summary.csv'),comparisons=read(ext/'paired_extension_comparisons.csv'),specialist_references=read(ext/'specialist_references.csv'),availability=read(ext/'baseline_availability.csv')))
    if len(result['comparisons'])!=39 or len(result['fusion_comparisons'])!=12:raise ValueError('H statistical registries incomplete')
    return result

def eligible_methods(cfg,groups):
    """Old confirmation runner evaluates every listed method at both budgets."""
    from src.week12.runner import MAIN
    expected={d:{r['query_id'] for r in rs} for d,rs in cohort(cfg).items()}
    pairs=[];excluded=[]
    for m in MAIN:
        complete=all(len(groups.get((d,m,b),[]))==len(ids) and {r['query_id'] for r in groups.get((d,m,b),[])}==ids for d,ids in expected.items() for b in cfg['budgets'])
        (pairs if complete else excluded).append([m,'BASE_FP16'])
    return pairs,excluded

def freeze_final(cfg,method,reason,retain_lora=False,dry_run=False):
    if method is None:return dict(status='REVIEW_REQUIRED',reason='Review H and explicitly select the preregistered fallback; no freeze written.')
    guard()
    if method!='FIXED_SENTENCE':raise ValueError('Reviewed H decision supports FIXED_SENTENCE fallback. Alternative promotion must pass the original noninferiority/Pareto gates; do not bypass them.')
    if retain_lora:raise ValueError('LoRA not run; RETAIN_LORA must remain False')
    if not isinstance(reason,str) or not reason.strip():raise ValueError('An evidence-based final rationale is required')
    result=load_saved_review(cfg)
    from src.week12.analysis import read_rows,confirmation_hash
    from src.week12.splits import read_freeze
    groups=defaultdict(list)
    for r in read_rows(cfg,False):
        if r['reader']=='BASE_FP16':groups[r['dataset'],r['method'],int(r['budget'])].append(r)
    pairs,excluded=eligible_methods(cfg,groups)
    if [method,'BASE_FP16'] not in pairs:raise ValueError('Fallback does not have complete frozen VAL coverage')
    decisions=json.loads((root(cfg)/'protocol/optional_phase_decisions.json').read_text())
    if any(decisions['phases'][p]['status']!='NOT_RUN_BY_USER_DECISION' for p in ['E','F','G']):raise ValueError('Reviewed optional phase decisions changed')
    split=read_freeze(cfg)
    evidence=result['comparisons'][(result['comparisons'].right=='FIXED_SENTENCE')&(result['comparisons'].reader=='BASE_FP16')]
    if len(evidence)!=6 or set(evidence.status)!={'COMPLETE'}:raise ValueError('Six complete matched QBGF/fixed-sentence comparisons required')
    obj=dict(method=method,selection_evidence=evidence.astype(object).where(pd.notna(evidence),None).to_dict('records'),rationale=reason,retain_lora=False,protocol=cfg,identity=identity(cfg),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),confirmation_ids_hash=confirmation_hash(cfg),timestamp=utc(),confirmation_evaluation_enabled=False,confirmation_history_review='LOCAL_ARTIFACT_AUDIT_COMPLETE_WITH_DOCUMENTED_LIMITATIONS',prompt_source_hash=sha(ROOT/'src/week12/reader.py'),phase='VALIDATION_ONLY',baseline_configurations=cfg,tokenizer=cfg['models']['Qwen'],reader='BASE_FP16',retriever='SFR',candidate_width=cfg['candidate_width'],budgets=cfg['budgets'],parser_hash=sha(ROOT/'src/week12/reader.py'),metrics=cfg['statistics'],adapter_hash=None,adapter_checkpoint=None,confirmation_methods=pairs,split_hash=split['split_hash'],confirmation_ids={d:[r['query_id'] for r in p['CONFIRM']] for d,p in split['datasets'].items()},optional_phase_decisions=decisions,excluded_unvalidated_confirmation_methods=excluded,confirmation_registry_rule='Retain existing MAIN controls only if complete BASE_FP16 VAL coverage at both budgets for every dataset. C2 at1024/unvalidated secondary controls excluded; qualified-only LLMLingua never enrolled. No new methods.',review_source_hash=sha(ROOT/'src/week12_final_review.py'),H_table_hashes={name:sha(root(cfg)/'analysis'/name) for name in ['summary.csv','primary_comparisons.csv','fusion_diagnostic_comparisons.csv']})
    # Keep exact compressor lock as provenance only, without enrolling it for confirmation.
    from src.week12.compression_runtime import status
    obj['llmlingua_effective_spec']=dict(status(cfg),quality_evaluation_status='NOT_RUN_BY_USER_DECISION',included_in_confirmation=False)
    if dry_run:return obj
    targets=[ROOT/'results/week12/W13_FINAL_METHOD_FREEZE.json',root(cfg)/'W13_FINAL_METHOD_FREEZE.json']
    # Resume byte-identical existing freeze; never replace its timestamp or decision.
    for p in targets:
        if p.exists():
            old=json.loads(p.read_text())
            for key in ['method','rationale','retain_lora','identity','split_hash','confirmation_methods','optional_phase_decisions','review_source_hash','H_table_hashes']:
                if old.get(key)!=obj[key]:raise ValueError('Existing final freeze differs; refuse overwrite: '+key)
            obj=old
    for p in targets:immutable(p,obj)
    return obj
