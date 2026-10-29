"""Saved development results only. Confirmation is never opened or used to select."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from .protocol import ROOT,root,Store,identity,immutable,atomic,digest,sha,utc
from .statistics import metric,paired,registry,holm,pareto,components,reduce

def read_rows(cfg,synthetic=False):
    prefix=root(cfg)/'smoke/synthetic'/identity(cfg)['code_hash'][:12] if synthetic else root(cfg)/('validation' if cfg.get('split_protocol') else 'development');rows=[]
    for p in prefix.glob('*/rows/identity.json'):rows+=Store(p.parent,json.loads(p.read_text())).rows()
    if not synthetic and cfg.get('split_protocol'):
        from .protocol import cohort
        allowed={d:{v['query_id'] for v in ids} for d,ids in cohort(cfg).items()}
        if any(r['query_id'] not in allowed[r['dataset']] or r.get('split')!='VALIDATION' for r in rows):raise ValueError('Analysis accepts frozen VAL rows only, never confirmation or historical pilot rows')
    dedup={}
    for r in rows:
        k=(r['dataset'],r['query_id'],r['method'],str(r['budget']),r['reader']);old=dedup.get(k)
        if old and any(old[x]!=r[x] for x in ('prediction','evidence_hash','source_hash')):raise ValueError('Duplicate phase outcomes disagree')
        dedup[k]=r
    return list(dedup.values())
def interaction(cells,name,n=10000,seed=42):
    maps=[{r['query_id']:r for r in cell} for cell in cells];qs=sorted(maps[0]);assert all(set(m)==set(qs) for m in maps)
    clusters=sorted({maps[0][q]['cluster_id'] for q in qs});lookup={c:i for i,c in enumerate(clusters)};vs=[]
    for m in maps:
        a=components([m[q] for q in qs],name);v=np.zeros((len(clusters),a.shape[1]))
        for i,q in enumerate(qs):v[lookup[m[q]['cluster_id']]]+=a[i]
        vs.append(v)
    coeff=[1,-1,-1,1];point=sum(c*float(reduce(v.sum(0),name)) for c,v in zip(coeff,vs));rng=np.random.default_rng(seed);samples=[]
    for start in range(0,n,256):
        ix=rng.integers(0,len(clusters),(min(256,n-start),len(clusters)));samples.extend(sum(c*reduce(v[ix].sum(1),name) for c,v in zip(coeff,vs)))
    lo,hi=np.quantile(samples,[.025,.975]);return dict(point=point,ci_low=float(lo),ci_high=float(hi),resamples=n,seed=seed)

def analyze(cfg,synthetic=False):
    rows=read_rows(cfg,synthetic);folder=root(cfg)/('smoke/analysis/'+identity(cfg)['code_hash'][:12] if synthetic else 'analysis');folder.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['method'],str(r['budget']),r['reader'])].append(r)
    summaries=[]
    for (dataset,method,budget,reader),g in sorted(groups.items()):
        name=cfg['statistics']['primary_metric'][dataset];summaries.append(dict(dataset=dataset,method=method,budget=budget,reader=reader,n_queries=len(g),quality=metric(g,name),metric=name,accuracy=metric(g,'accuracy') if dataset=='SciFact' else None,HELP=np.mean([r['HELP'] for r in g]),HURT=np.mean([r['HURT'] for r in g]),NEUTRAL=np.mean([r['NEUTRAL'] for r in g]),answer_changed=np.mean([r['answer_changed'] for r in g]),evidence_tokens=np.mean([r['evidence_tokens'] for r in g]),prompt_tokens=np.mean([r['prompt_tokens'] for r in g]),unused_budget=np.mean([r['unused_budget'] for r in g if r['unused_budget'] is not None]) if any(r['unused_budget'] is not None for r in g) else None,budget_utilization=np.mean([r['budget_utilization'] for r in g if r['budget_utilization'] is not None]) if any(r['budget_utilization'] is not None for r in g) else None,generated_tokens=np.mean([r['generated_tokens'] for r in g]),conceptual_lm_calls=1,compression_ratio=np.mean([r['compression_ratio'] for r in g if r['compression_ratio'] is not None]) if any(r['compression_ratio'] is not None for r in g) else None,format_compliance=np.mean([r['format_compliant'] for r in g]) if dataset=='SciFact' else None,invalid_outputs=sum(r['prediction']=='INVALID_OUTPUT' for r in g),synthetic=synthetic))
    frame=pd.DataFrame(summaries);frame.to_csv(folder/'summary.csv',index=False)
    specs=registry(cfg,llm=True,lora=True);immutable(root(cfg)/'protocol'/'primary_comparisons.json',specs);comparisons=[];raw=[]
    from .protocol import cohort
    expected={d:1 for d in cfg['datasets']} if synthetic else {d:len(v) for d,v in cohort(cfg).items()}
    for spec in specs:
        a=groups.get((spec['dataset'],spec['left'],str(spec['budget']),spec['reader']),[]);b=groups.get((spec['dataset'],spec['right'],str(spec['budget']),spec.get('right_reader',spec['reader'])),[])
        ok=len(a)==len(b)==expected[spec['dataset']]
        if ok:result=paired(a,b,spec['metric'],cfg['bootstrap_resamples'],cfg['seed']);raw.append(result['raw_p']);comparisons.append(dict(spec,status='COMPLETE',**result))
        else:raw.append(1.);comparisons.append(dict(spec,status='PENDING_INCOMPLETE_OR_UNQUALIFIED',raw_p=None,point=None,ci_low=None,ci_high=None))
    adjusted=holm(raw)
    for c,p in zip(comparisons,adjusted):c['holm_adjusted_p']=p if c['status']=='COMPLETE' else None;c['family_size']=len(specs)
    from .statistics import fusion_registry
    fusion_specs=fusion_registry(cfg);immutable(root(cfg)/'protocol/fusion_diagnostic_comparisons.json',fusion_specs)
    fusion=[];fusion_raw=[]
    for spec in fusion_specs:
        a=groups.get((spec['dataset'],spec['left'],str(spec['budget']),spec['reader']),[]);b=groups.get((spec['dataset'],spec['right'],str(spec['budget']),spec['reader']),[])
        if len(a)==len(b)==expected[spec['dataset']]:
            result=paired(a,b,spec['metric'],cfg['bootstrap_resamples'],cfg['seed']);fusion_raw.append(result['raw_p']);fusion.append(dict(spec,status='COMPLETE',**result))
        else:fusion_raw.append(1.);fusion.append(dict(spec,status='PENDING_INCOMPLETE',raw_p=None,point=None,ci_low=None,ci_high=None))
    # Preserve original primary-family correction; additionally guard joint claims across all 51 contrasts.
    joint=holm(raw+fusion_raw)
    for row,pvalue in zip(comparisons+fusion,joint):
        row['joint_family_size']=len(joint);row['joint_holm_adjusted_p']=pvalue if row['status']=='COMPLETE' else None
    pd.DataFrame(fusion).to_csv(folder/'fusion_diagnostic_comparisons.csv',index=False)
    paired_df=pd.DataFrame(comparisons);paired_df.to_csv(folder/'primary_comparisons.csv',index=False)
    perclass=[]
    for (dataset,method,budget,reader),g in groups.items():
        if dataset!='SciFact':continue
        for label in cfg['labels']:
            tp=sum(r['gold']==label and r['prediction']==label for r in g);fp=sum(r['gold']!=label and r['prediction']==label for r in g);fn=sum(r['gold']==label and r['prediction']!=label for r in g)
            perclass.append(dict(dataset=dataset,method=method,budget=budget,reader=reader,label=label,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,support=sum(r['gold']==label for r in g)))
    pd.DataFrame(perclass).to_csv(folder/'scifact_per_class.csv',index=False)
    evidence=[]
    for k,g in groups.items():
        evidence.append(dict(dataset=k[0],method=k[1],budget=k[2],reader=k[3],mean_any_gold_recall=np.mean([r['any_gold_recall'] for r in g if r['any_gold_recall'] is not None]) if any(r['any_gold_recall'] is not None for r in g) else None,valid_gold_queries=sum(r['any_gold_recall'] is not None for r in g),unmapped_gold=sum(r.get('unmapped_gold') or 0 for r in g)))
    pd.DataFrame(evidence).to_csv(folder/'evidence_metrics.csv',index=False)
    # Reserved optional hypotheses count in Holm at p=1 internally, never reported as observed tests.
    allocation=[]
    for r in rows:
        for choice in r['choices']:allocation.append(dict(dataset=r['dataset'],query_id=r['query_id'],method=r['method'],budget=r['budget'],reader=r['reader'],**choice))
    pd.DataFrame(allocation).to_csv(folder/'allocation.csv',index=False)
    frontier=pd.DataFrame(pareto(summaries));frontier.to_csv(folder/'pareto.csv',index=False)
    lora=[]
    for dataset in cfg['datasets']:
        for budget in cfg['budgets']:
            cells=[groups.get((dataset,mode,str(budget),reader),[]) for mode,reader in [('QBGF_FULL','LORA_NF4'),('QBGF_FULL','BASE_NF4'),('RAW_SKIP','LORA_NF4'),('RAW_SKIP','BASE_NF4')]]
            if all(len(c)==expected[dataset] for c in cells):
                values=[metric(c,cfg['statistics']['primary_metric'][dataset]) for c in cells]
                lora.append(dict(dataset=dataset,budget=budget,selection_effect=values[1]-values[3],adaptation_RAW=values[2]-values[3],adaptation_QBGF=values[0]-values[1],combined=values[0]-values[3],interaction=values[0]-values[1]-values[2]+values[3],**{'interaction_'+k:v for k,v in interaction(cells,cfg['statistics']['primary_metric'][dataset],cfg['bootstrap_resamples']).items()}))
    pd.DataFrame(lora).to_csv(folder/'lora_decomposition.csv',index=False)
    # Source tables and deterministic plot scaffolding; absent methods stay absent.
    figures=[]
    if not frame.empty:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        views={'quality_budget':None,'quality_prompt_tokens':None,'pareto':None,'representation_choices':['QBGF_FULL'],'budget_utilization':None,'query_conditioning':['QBGF_FULL','QBGF_NO_QUERY'],'raw_strict_skip':['RAW_STRICT','RAW_SKIP'],'qbgf_sentence':['QBGF_FULL','FIXED_SENTENCE'],'qbgf_h3':['QBGF_FULL','H3'],'llmlingua':['QBGF_FULL','LLMLINGUA2'],'lora_2x2':['QBGF_FULL','RAW_SKIP'],'help_hurt':None}
        for dataset in cfg['datasets']:
            for name,methods in views.items():
                g=frame[frame.dataset==dataset];g=g[g.method.isin(methods)] if methods else g
                if name=='llmlingua' and 'LLMLINGUA2' not in set(g.method):continue
                if name=='lora_2x2' and 'LORA_NF4' not in set(g.reader):continue
                path=folder/'figures'/f'{dataset}_{name}';path.parent.mkdir(parents=True,exist_ok=True)
                if name=='representation_choices':
                    a=pd.DataFrame(allocation);g=a[(a.dataset==dataset)&(a.method=='QBGF_FULL')] if not a.empty else a
                    if g.empty:continue
                    source=g.groupby(['budget','name']).size().reset_index(name='count');fig,ax=plt.subplots(figsize=(7,4));ax.bar([str(b)+'/'+m for b,m in zip(source.budget,source.name)],source['count']);ax.set_ylabel('Selected options, including DROP')
                else:
                    source=g;fig,ax=plt.subplots(figsize=(7,4));x='prompt_tokens' if name in ('quality_prompt_tokens','pareto') else 'budget' if name=='quality_budget' else 'evidence_tokens';y='HELP' if name=='help_hurt' else 'budget_utilization' if name=='budget_utilization' else 'quality';
                    for (method,reader),h in g.groupby(['method','reader']):
                        h=h.copy()
                        if x=='budget':h[x]=pd.to_numeric(h[x],errors='coerce');h=h.dropna(subset=[x]);h=h.sort_values(x)
                        if h.empty:continue
                        ax.scatter(h[x],h[y],label=method+'/'+reader)
                        if name=='help_hurt':ax.scatter(h[x],h['HURT'],marker='x',label=method+'/'+reader+' HURT')
                    ax.set_xlabel(x);ax.set_ylabel(y);ax.legend(fontsize=6)
                source.to_csv(path.with_suffix('.csv'),index=False);ax.set_title(('SYNTHETIC SMOKE — ' if synthetic else 'Development descriptive — ')+dataset+' '+name);fig.tight_layout();fig.savefig(path.with_suffix('.png'));fig.savefig(path.with_suffix('.pdf'));plt.close(fig);figures.append(dict(dataset=dataset,name=name,png=str(path.with_suffix('.png')),pdf=str(path.with_suffix('.pdf')),data=str(path.with_suffix('.csv'))))
    complete=paired_df[paired_df.status=='COMPLETE']
    if not complete.empty:
        import matplotlib.pyplot as plt
        for d,g in complete.groupby('dataset'):
            fig,ax=plt.subplots(figsize=(9,max(3,len(g)*.3)));yy=np.arange(len(g));ax.errorbar(g.point,yy,xerr=[g.point-g.ci_low,g.ci_high-g.point],fmt='o');ax.axvline(0,color='gray');ax.set_yticks(yy,g.id,fontsize=6);ax.set_xlabel('Paired primary metric difference; nominal 95% cluster bootstrap CI');ax.set_title(('SYNTHETIC SMOKE — ' if synthetic else 'Development — ')+d);fig.tight_layout();p=folder/'figures'/f'{d}_primary_intervals';g.to_csv(p.with_suffix('.csv'),index=False);fig.savefig(p.with_suffix('.png'));fig.savefig(p.with_suffix('.pdf'));plt.close(fig);figures.append(dict(dataset=d,name='primary_intervals',png=str(p.with_suffix('.png')),pdf=str(p.with_suffix('.pdf')),data=str(p.with_suffix('.csv'))))
    references=[]
    if not synthetic:
        for p in (root(cfg)/'canonical_references').glob('*/identity.json'):references+=Store(p.parent,json.loads(p.read_text())).rows()
    types=[]
    for key,g in groups.items():
        for answer_type in sorted({str(r.get('answer_type','unknown')) for r in g}):
            sub=[r for r in g if str(r.get('answer_type','unknown'))==answer_type]
            types.append(dict(dataset=key[0],method=key[1],budget=key[2],reader=key[3],answer_type=answer_type,n_queries=len(sub),quality=metric(sub,cfg['statistics']['primary_metric'][key[0]])))
    pd.DataFrame(types).to_csv(folder/'answer_type_breakdown.csv',index=False)
    reference_frame=pd.DataFrame(references);reference_frame.to_csv(folder/'specialist_references.csv',index=False)
    atomic(folder/'figures.json',figures);atomic(folder/'status.json',dict(saved_result_analysis_only=True,synthetic=synthetic,n_rows=len(rows),n_conditions=len(summaries),primary_family_size=len(specs),no_confirmation_outcomes=True))
    (folder/'REPORT.md').write_text('# Week12 '+('SYNTHETIC SMOKE ONLY' if synthetic else 'development analysis')+'\n\n'+(frame.to_markdown(index=False) if not frame.empty else 'No completed rows. No results fabricated.')+'\n\nHolm family is fixed across datasets/budgets, including reserved optional arms at p=1 internally. Missing tests are pending, never observed p=1 claims. Macro-F1 is recomputed per cluster resample. Prompt/evidence Pareto comparisons stay within reader precision. End-to-end latency is unmeasured.\n')
    return dict(summary=frame,comparisons=paired_df,fusion_comparisons=pd.DataFrame(fusion),specialist_references=reference_frame,figures=figures,folder=str(folder))

def freeze_final(cfg,method,reason,retain_lora=False):
    if method is None:return dict(status='REVIEW_REQUIRED',reason='No automatic winner selection. Supply method and evidence-based rationale after development analysis.')
    if method not in ['QBGF_FULL','FIXED_SENTENCE','C2','H3','RAW_SKIP']:raise ValueError('Unsupported final policy')
    if not reason:raise ValueError('Explicit review rationale required')
    base=root(cfg)/('validation' if cfg.get('split_protocol') else 'development')
    if not (base/'C/status.json').exists():raise ValueError('Full development results required, never freeze synthetic smoke')
    s=json.loads((base/'C/status.json').read_text());assert s['complete'] and s['full_generation']
    if retain_lora and method!='QBGF_FULL':raise ValueError('Only the registered QBGF LoRA policy has a matched 2x2 evaluation')
    if retain_lora and not (root(cfg)/'lora/complete.json').exists():raise ValueError('Cannot retain untrained LoRA')
    groups={}
    for r in read_rows(cfg,False):groups.setdefault((r['dataset'],r['method'],str(r['budget']),r['reader']),[]).append(r)
    from .protocol import cohort
    expected_ids={d:{r['query_id'] for r in rows} for d,rows in cohort(cfg).items()}
    for d,ids in expected_ids.items():
        for b in cfg['budgets']:
            if {r['query_id'] for r in groups.get((d,method,str(b),'BASE_FP16'),[])}!=ids:raise ValueError('Final selection requires the entire frozen VAL split, including for the simple baseline')
    decision=[];any_cost_gain=False
    if method!='FIXED_SENTENCE':
        for d in cfg['datasets']:
            for b in cfg['budgets']:
                a=groups.get((d,method,str(b),'BASE_FP16'),[]);z=groups.get((d,'FIXED_SENTENCE',str(b),'BASE_FP16'),[])
                if {r['query_id'] for r in a}!={v['query_id'] for v in __import__('src.week12.protocol',fromlist=['cohort']).cohort(cfg)[d]} or {r['query_id'] for r in z}!={r['query_id'] for r in a}:raise ValueError('Complete paired development surface required for final selection')
                ci=paired(a,z,cfg['statistics']['primary_metric'][d],cfg['bootstrap_resamples'],cfg['seed']);margin=cfg['selection_rule']['quality_margin_macro_f1' if d=='SciFact' else 'quality_margin_qa']
                if ci['ci_low'] < -margin:raise ValueError('Preregistered noninferiority gate not met; retain simpler sentence policy')
                token_delta=sum(r['prompt_tokens'] for r in a)/len(a)-sum(r['prompt_tokens'] for r in z)/len(z)
                if token_delta>0:raise ValueError('Candidate is not token-Pareto admissible versus fixed sentences')
                any_cost_gain=any_cost_gain or token_delta<0;decision.append(dict(dataset=d,budget=b,token_delta=token_delta,margin=margin,**ci))
        if not any_cost_gain:raise ValueError('No token advantage established; retain simpler policy')
    if retain_lora:
        supported_gain=False
        for d in cfg['datasets']:
            for b in cfg['budgets']:
                adapted=groups.get((d,'QBGF_FULL',str(b),'LORA_NF4'),[]);base_rows=groups.get((d,'QBGF_FULL',str(b),'BASE_NF4'),[])
                if not adapted or not base_rows:raise ValueError('Complete VAL 2x2 required for LoRA retention')
                ci=paired(adapted,base_rows,cfg['statistics']['primary_metric'][d],cfg['bootstrap_resamples'],cfg['seed'])
                margin=cfg['selection_rule']['quality_margin_macro_f1' if d=='SciFact' else 'quality_margin_qa']
                if ci['ci_low'] < -margin:raise ValueError('VAL cannot rule out LoRA harm; do not retain adapter')
                supported_gain=supported_gain or ci['ci_low']>0
        if not supported_gain:raise ValueError('No paired VAL gain supports retaining LoRA')
        gstatus=base/'G/status.json'
        if not gstatus.exists() or not json.loads(gstatus.read_text()).get('complete'):raise ValueError('Mandatory real LoRA 2x2 must be complete before retaining adapter')
    import subprocess
    obj=dict(method=method,selection_evidence=decision,rationale=reason,retain_lora=retain_lora,protocol=cfg,identity=identity(cfg),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),confirmation_ids_hash=confirmation_hash(cfg),timestamp=utc(),confirmation_evaluation_enabled=False,confirmation_history_review='REQUIRED_BEFORE_WEEK13',prompt_source_hash=sha(ROOT/'src/week12/reader.py'))
    obj.update(phase='VALIDATION_ONLY',baseline_configurations=cfg,tokenizer=cfg['models']['Qwen'],reader='LORA_NF4' if retain_lora else 'BASE_FP16',retriever='SFR',candidate_width=cfg['candidate_width'],budgets=cfg['budgets'],parser_hash=sha(ROOT/'src/week12/reader.py'),metrics=cfg['statistics'],adapter_hash=json.loads((root(cfg)/'lora/complete.json').read_text())['adapter_hash'] if retain_lora else None,adapter_checkpoint=str(root(cfg)/'lora/adapter') if retain_lora else None)
    from .runner import MAIN
    pairs=[(m,'BASE_FP16') for m in MAIN]
    qual_path=root(cfg)/'qualification/llmlingua.json'
    from .compression_runtime import status as compressor_status
    compressor_lock=compressor_status(cfg)
    obj['llmlingua_effective_spec']=compressor_lock
    if compressor_lock['status']=='QUALIFIED':pairs.append(('LLMLINGUA2','BASE_FP16'))
    if retain_lora:pairs += [(m,r) for m in ['RAW_SKIP','QBGF_FULL'] for r in ['BASE_NF4','LORA_NF4']]
    obj['confirmation_methods']=pairs
    if cfg.get('split_protocol'):
        from .splits import read_freeze
        freeze=read_freeze(cfg);obj.update(split_hash=freeze['split_hash'],confirmation_ids={d:[r['query_id'] for r in p['CONFIRM']] for d,p in freeze['datasets'].items()},confirmation_history_review='LOCAL_ARTIFACT_AUDIT_COMPLETE_WITH_DOCUMENTED_LIMITATIONS')
        immutable(ROOT/'results/week12/W13_FINAL_METHOD_FREEZE.json',obj)
    return immutable(root(cfg)/'W13_FINAL_METHOD_FREEZE.json',obj)

def confirmation_hash(cfg):
    if cfg.get('split_protocol'):
        from .splits import read_freeze
        return digest({d:p['CONFIRM'] for d,p in read_freeze(cfg)['datasets'].items()})
    return json.loads((root(cfg)/'protocol/W12_CONFIRMATION_FREEZE.json').read_text())['ids_hash']
