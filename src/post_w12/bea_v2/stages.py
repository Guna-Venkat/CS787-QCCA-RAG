"""CPU cache-based calibration/preparation; manual answer inference only."""
import shutil,time
import numpy as np
import pandas as pd
from .protocol import *
from .data import problem,feature_rows
from .models import train,predict,verify_models
from .allocator import allocate
from src.post_w12.bea_qbgf.allocator import behavior

def tokenizer(parent):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(parent['models']['Qwen']['id'],revision=parent['models']['Qwen']['revision'],local_files_only=True)

def allocation(c,v1,member,d,b,tok,threshold):
    q=member['query_id'];p=problem(c,v1,d,q);ph,pm,meta=predict(c,d,q,member['group'],feature_rows(p,b))
    sel=allocate(p['pool'],p['sentences'],p['scores'],dict(p_help=ph,p_harm=pm),tok,b,c['parameters'],threshold,c['source_states'])
    from src.week12.algorithm import serialize
    check=serialize(sel['units'],tok)
    if check['evidence_ids']!=sel['evidence_ids'] or check['evidence_tokens']!=sel['evidence_tokens']:raise ValueError('Exact serializer mismatch')
    for u in sel['units']:
        original=next((x for x in p['pool']+p['sentences'] if x['unit_id']==u['unit_id']),None)
        if original is None or original['source_text']!=u['source_text'] or original['hash']!=u['hash']:raise ValueError('Selected source changed')
    sel['lineage']=p['lineage'];bh=behavior(sel,p['sentences'],p['scores'])
    terms=[s for t in sel['trace'] for s in t['sentences']]
    chosen_ids={i for i,u in enumerate(p['sentences']) if u['unit_id'] in sel['selected_unit_ids'] or u['passage_id'] in {x['unit_id'] for x in sel['units'] if x['unit_type']=='PASSAGE'}}
    bh.update(source_expansion=any(v>=2 for v in sel['source_positive_contributions'].values()),
              mean_selected_p_help=float(np.mean([ph[i] for i in chosen_ids])) if chosen_ids else None,
              mean_selected_p_harm=float(np.mean([pm[i] for i in chosen_ids])) if chosen_ids else None,
              mean_selected_D=float(np.mean([ph[i] for i in chosen_ids])) if chosen_ids else None,
              mean_help_minus_harm_diagnostic=float(np.mean([ph[i]-pm[i] for i in chosen_ids])) if chosen_ids else None,
              mean_transition_IG=float(np.mean([s['ig'] for s in terms])) if terms else None,
              mean_transition_U=float(np.mean([s['value'] for s in terms])) if terms else None)
    return dict(selection=sel,behavior=bh,source_hash=p['source_hash'],predictor=meta)

def calibration_summary(frame,c):
    """Outcome-free preregistered global rule plus every-cell anti-collapse checks."""
    cfg=c['calibration'];base={(r['dataset'],r['query_id'],r['budget']):r['coverage'] for r in frame.to_dict('records') if r['threshold']==0}
    expected={(d,b) for d in DATASETS for b in c['budgets']};cells=[];summary=[]
    for tau,xs in frame.groupby('threshold'):
        if {(r['dataset'],r['budget']) for r in xs.to_dict('records')}!=expected:raise ValueError('Incomplete calibration cells')
        retention=[];cell_checks=[]
        for (d,b),group in xs.groupby(['dataset','budget']):
            if len(group)!=cfg['queries_per_dataset'] or group.query_id.nunique()!=len(group):raise ValueError('Wrong unique-query calibration count')
            ratios=[r['coverage']/base[r['dataset'],r['query_id'],r['budget']] if base[r['dataset'],r['query_id'],r['budget']]>0 else 1. for r in group.to_dict('records')]
            retention.extend(ratios)
            cell=dict(threshold=float(tau),dataset=d,budget=int(b),n_queries=len(group),
                      empty_contexts=int((group.units==0).sum()),mean_utilization=float(group.utilization.mean()),
                      median_units=float(group.units.median()),trivial_fraction=float(((group.utilization<cfg['trivial_utilization_below'])|(group.units<2)).mean()),
                      coverage_retention=float(np.mean(ratios)),threshold_stop_fraction=float((group.stop_reason=='LOW_MARGINAL_VALUE').mean()))
            failures=[]
            if cell['empty_contexts']:failures.append('empty_context')
            if cell['mean_utilization']<cfg['minimum_cell_mean_utilization']:failures.append('low_mean_utilization')
            if cell['median_units']<cfg['minimum_cell_median_units']:failures.append('low_median_unit_count')
            if cell['trivial_fraction']>cfg['maximum_cell_trivial_fraction']+1e-12:failures.append('too_many_trivial_contexts')
            if cell['coverage_retention']<cfg['minimum_cell_coverage_retention']:failures.append('coverage_loss')
            cell.update(eligible=not failures,failure_reasons=';'.join(failures));cells.append(cell);cell_checks.append(cell['eligible'])
        r=dict(threshold=float(tau),n=len(xs),threshold_stop_fraction=float((xs.stop_reason=='LOW_MARGINAL_VALUE').mean()),
               budget_stop_fraction=float((xs.stop_reason=='BUDGET_EXHAUSTED').mean()),mean_utilization=float(xs.utilization.mean()),
               near_full_fraction=float((xs.utilization>=.95).mean()),empty_fraction=float((xs.units==0).mean()),coverage_retention=float(np.mean(retention)),
               all_cells_eligible=all(cell_checks),failed_cells=sum(not x for x in cell_checks))
        r['eligible']=bool(r['threshold_stop_fraction']>=cfg['minimum_threshold_stop_fraction'] and r['near_full_fraction']<=cfg['maximum_near_full_fraction'] and r['empty_fraction']==0 and r['coverage_retention']>=cfg['minimum_coverage_retention'] and r['all_cells_eligible'])
        summary.append(r)
    return summary,cells

@exclusive
def calibrate(c=None):
    c=c or load();models=verify_models(c);parent,v1,s,cohort=dependencies(c);out=root(c)/'calibration';receipt=out/'CALIBRATION.json'
    ident=dict(config=c,implementation=implementation(),model_manifest_sha=sha(root(c)/'utility_model/MODELS.json'))
    if receipt.exists():
        value=read(receipt)
        if value['identity_hash']!=digest(ident):raise ValueError('Calibration inputs changed')
        return value
    small={d:sorted(rows,key=lambda r:r['query_id'])[:c['calibration']['queries_per_dataset']] for d,rows in cohort['selected'].items()}
    if sum(map(len,small.values()))!=9:raise ValueError('This version preregisters nine unique calibration queries')
    write(out/'PREREGISTRATION.json',dict(identity=ident,identity_hash=digest(ident),cohort=small,rule=c['calibration'],formulation=c['formulation'],answer_outcomes_accessed=False))
    write(out/'COHORT.json',small);tok=tokenizer(parent);st=Store(out/'allocation_traces',ident);rows=[];dist=[]
    for d,members in small.items():
        for member in members:
            for b in c['budgets']:
                for tau in c['calibration']['threshold_grid']:
                    key=[d,member['query_id'],b,tau];item=st.get(key)
                    if item is None:item=st.put(key,allocation(c,v1,member,d,b,tok,tau))
                    sel=item['selection'];bh=item['behavior']
                    rows.append(dict(dataset=d,query_id=member['query_id'],budget=b,threshold=tau,
                                     stop_reason=sel['stop_reason'],utilization=sel['evidence_tokens']/b,units=len(sel['units']),
                                     represented_parents=bh['represented_parents'],coverage=bh['aspect_coverage'],full_sources=sel['full_sources']))
                    for step in sel['marginal_distribution']:
                        for value in step['values']:dist.append(dict(dataset=d,query_id=member['query_id'],budget=b,threshold=tau,step=step['step'],marginal=value))
            print('CPU allocation smoke complete:',d,member['query_id'],'(no answer generation)',flush=True)
    frame=pd.DataFrame(rows);summary,cells=calibration_summary(frame,c)
    text(out/'cell_checks.csv',pd.DataFrame(cells).to_csv(index=False))
    choices=[r['threshold'] for r in summary if r['eligible']];chosen=min(choices) if choices else None
    for name,df in [('allocations',frame),('threshold_summary',pd.DataFrame(summary)),('marginal_distribution',pd.DataFrame(dist))]:text(out/(name+'.csv'),df.to_csv(index=False))
    value=dict(status='READY' if choices else 'NEEDS_REVIEW',threshold=chosen,identity_hash=digest(ident),model_manifest_sha=ident['model_manifest_sha'],answer_generations=0,queries=9,quality_or_heldout_LOO_labels_used=False,summary=summary,cell_checks=cells,preregistration_sha=sha(out/'PREREGISTRATION.json'))
    write(receipt,value)
    text(out/'ALLOCATION_CALIBRATION_REPORT.md','# Nine-query CPU allocation-only calibration\n\nThree frozen development queries per dataset; both budgets. No reader generation, F1, gold evidence or held-out LOO labels enter allocation/calibration. Predictor probabilities come from each query/group’s held-out model. P_harm is diagnostic-only. The exact sample/grid and global/every-cell anti-collapse rule were saved before the first allocation in PREREGISTRATION.json.\n\n'+pd.DataFrame(summary).to_markdown(index=False)+'\n\n## Every dataset/budget cell\n\n'+pd.DataFrame(cells).to_markdown(index=False)+'\n\nStatus: '+value['status']+'; threshold: '+str(chosen)+'. No post-hoc relaxation is allowed. NEEDS_REVIEW blocks the full pilot. These are calibration checks, not forced minimum-budget constraints or answer-quality evidence.\n')
    return value

@exclusive
def prepare(c=None):
    c=c or load();parent,v1,s,cohort,f=freeze(c);tok=tokenizer(parent);store=Store(root(c)/'pilot/selections',f['identity'])
    for d,members in cohort['selected'].items():
        for member in members:
            for b in c['budgets']:
                key=[d,member['query_id'],'BEA_V2',b]
                if store.get(key) is None:store.put(key,allocation(c,v1,member,d,b,tok,f['identity']['threshold']))
            print('Prepared/reused V2',d,member['query_id'],flush=True)

def generation_key(e,sel,parent):
    from src.week12.reader import prompt_ids
    return [e['dataset'],e['query_id'],digest(prompt_ids(e['_tok'],e['question'],e['task_type'],sel['evidence_ids'],parent['labels']))]

@exclusive
def generate(c=None):
    c=c or load();parent,v1,s,cohort,f=freeze(c)
    from src.post_w12.bea_qbgf.resources import load_val
    from src.week12.reader import Reader,prompt_ids
    from src.week11.evaluation import evaluate
    tok=tokenizer(parent);xs={(e['dataset'],e['query_id']):e for e in load_val(parent,s,cohort)}
    selections=Store(root(c)/'pilot/selections',f['identity']);results=Store(root(c)/'pilot/rows',f['identity']);gen=Store(root(c)/'cache/generations',f['identity'])
    reader=None;receipts=[];start=time.time()
    try:
        for d,members in cohort['selected'].items():
            for member in members:
                q=member['query_id'];e=xs[d,q]
                for b in c['budgets']:
                    key=[d,q,'BEA_V2',b];row=results.get(key)
                    if row is not None:receipts.append(dict(dataset=d,query_id=q,budget=b,cache_source='V2 completed row',physical_generation_performed=False));continue
                    item=selections.get(key)
                    if item is None:raise ValueError('Run missing V2 preparation before generation')
                    sel=item['selection']
                    if item['source_hash']!=digest(e['inference']):raise ValueError('Current source differs from frozen cached problem')
                    expected=digest(prompt_ids(tok,e['question'],e['task_type'],sel['evidence_ids'],parent['labels']));gkey=[d,q,expected]
                    answer=gen.get(gkey);source='V2 exact-prompt cache';physical=False
                    if answer is None and c['cache']['reuse_compatible']:
                        for oldm in ['FIXED_SENTENCE','QBGF_FULL','BEA_QBGF']:
                            for oldb in c['budgets']:
                                path=old(c)/'cache/results'/(digest([d,q,oldm,oldb])+'.json')
                                if not path.exists():continue
                                candidate=legacy(c,'cache/results',[d,q,oldm,oldb],v1['identity_hash'])
                                if candidate['prompt_hash']==expected and candidate['source_hash']==item['source_hash']:
                                    fields=['prediction','raw_prediction','reader','model_hash','prompt_hash','prompt_tokens','generated_tokens','generation_seconds','generation_cap_reached']
                                    answer={k:candidate[k] for k in fields if k in candidate};source='historical exact-prompt '+oldm;break
                            if answer is not None:break
                    if answer is None:
                        if shutil.disk_usage(root(c)).free<c['minimum_free_disk_gib']*1024**3:raise RuntimeError('Disk headroom insufficient; completed rows resumable')
                        if reader is None:
                            import torch
                            if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<18*1024**3:raise RuntimeError('Manual pilot requires a free GPU for frozen BASE_FP16. Do not interrupt V1; wait until it finishes.')
                            reader=Reader(parent,'BASE_FP16')
                        answer=reader.generate(e,sel);physical=True;source='new frozen BASE_FP16 answer'
                    if answer['prompt_hash']!=expected or answer['reader']!='BASE_FP16' or answer['model_hash']!=digest([parent['models']['Qwen'],'BASE_FP16',None]):raise ValueError('Wrong reader/prompt identity')
                    gen.put(gkey,answer);metrics=evaluate(answer['prediction'],e)
                    row=dict(label=LABEL,study=c['study'],dataset=d,query_id=q,paper_id=e['paper_id'],cluster_id=member['group'],split='VAL',method='BEA_V2',budget=b,
                             **metrics,**answer,gold=e['evaluation']['gold_label'] if d=='SciFact' else e['evaluation']['gold_answer'],
                             correct=int(answer['prediction']==e['evaluation']['gold_label']) if d=='SciFact' else None,
                             evidence_tokens=sel['evidence_tokens'],unused_budget=b-sel['evidence_tokens'],budget_utilization=sel['evidence_tokens']/b,
                             evidence_hash=sel['evidence_hash'],evidence_ids=sel['evidence_ids'],evidence_text=sel['text'],selected_unit_ids=sel['selected_unit_ids'],
                             behavior=item['behavior'],stop_reason=sel['stop_reason'],source_hash=item['source_hash'],predictor=item['predictor'],
                             cache_source=source,physical_generation_performed=physical,conceptual_lm_calls=1,identity_hash=f['identity_hash'])
                    results.put(key,row);receipts.append(dict(dataset=d,query_id=q,budget=b,cache_source=source,physical_generation_performed=physical))
                    write(root(c)/'pilot/receipts'/(digest(key)+'.json'),receipts[-1])
                    print(f'V2 durable row {len(receipts)}/300; elapsed={time.time()-start:.0f}s',flush=True)
    finally:
        if reader is not None:reader.close()
        if receipts:text(root(c)/'pilot/cache_manifest.csv',pd.DataFrame(receipts).to_csv(index=False))
    write(root(c)/'pilot/complete.json',dict(identity_hash=f['identity_hash'],logical_rows=300))
