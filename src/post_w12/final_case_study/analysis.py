"""Manual CPU-only retrospective dossiers from frozen saved results.

No allocator, learned predictor, encoder, reader or answer generator is invoked.
Gold is read ONLY by this diagnostic analysis, never fed back into allocation.
"""
import collections
import contextlib
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from .gold import annotations,retention,retrieval_status
from .selection import METHODS,PREFIXES,PAIRS,RULES,choose

LABEL='POST_W12_RETROSPECTIVE_QUALITATIVE_CASE_STUDY'
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/post_w12/final_case_study'
DATASETS=['HotpotQA','QASPER','SciFact']
LABELS=['RETRIEVAL_MISS','SELECTION_MISS','OVER_PRUNING','OVER_DIVERSIFICATION',
 'REDUNDANCY_OVERPENALIZATION','CONTEXT_FRAGMENTATION','CONTEXT_DILUTION',
 'FULL_CONTEXT_RECOVERY','COMPLEMENTARY_EVIDENCE_RECOVERY','PHELP_MISCALIBRATION',
 'THRESHOLD_MISCALIBRATION','READER_REASONING_FAILURE','PARSER_OR_METRIC_EFFECT','AMBIGUOUS']
NOTICE=('Retrospective, outcome-selected explanatory case studies, not confirmation or '
 'population failure frequencies. Gold annotations are diagnostic only and were not '
 'available to the allocator. No allocation, predictor, threshold or answer is changed. '
 'The BEA_V2 preregistered calibration remains CALIBRATION_FAILED_TERMINATED; '
 'the V2 answers come from the separately labelled post-hoc tau=8e-5 study.')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def save(path,value,immutable=False):
    path=Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):raise ValueError('Only case-study outputs are writable')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not isinstance(value,str):value=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if immutable and path.exists():
        if path.read_text()!=value:raise ValueError('Incompatible analysis identity; preserve previous outputs')
        return
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.pending-')
    try:
        with os.fdopen(fd,'w') as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def csv(name,values,columns=None):
    frame=pd.DataFrame(values,columns=columns)
    save(OUT/name,frame.to_csv(index=False));return frame

def diagnostic_units(row,p,sel,tok,e):
    from src.week12.algorithm import serialize
    index={u['unit_id']:u for u in p['pool']+p['sentences']}
    ids=row['selected_unit_ids']
    if not set(ids)<=index.keys():raise ValueError('Unknown saved selected unit')
    units=[index[i] for i in ids]
    docs={d['document_id']:d['text'] for d in e['inference']['documents']}
    for u in units:
        if docs[u['document_id']][u['start_char']:u['end_char']]!=u['source_text']:
            raise ValueError('Selected sentence/source offsets differ')
    check=serialize(units,tok)
    if (check['evidence_ids']!=row['evidence_ids'] or check['text']!=row['evidence_text']
        or check['evidence_tokens']!=row['evidence_tokens']):
        raise ValueError('Saved context cannot be reproduced by the frozen serializer')
    if sel is not None and sel['selection']['evidence_ids']!=row['evidence_ids']:
        raise ValueError('Saved trace and saved answer used different evidence')
    return check['units']

def structure(units,p,sel):
    parents={u['passage_id'] for u in units};full={u['unit_id'] for u in units if u['unit_type']=='PASSAGE'}
    covered={s['unit_id'] for s in p['sentences'] if s['unit_id'] in {u['unit_id'] for u in units} or s['passage_id'] in full}
    allby=collections.defaultdict(list)
    for s in p['sentences']:allby[s['passage_id']].append(s)
    adjacency=0;possible=0;runs=0
    for ss in allby.values():
        ss=sorted(ss,key=lambda s:(s['start_char'],s['unit_id']));previous=False
        for i,s in enumerate(ss):
            active=s['unit_id'] in covered
            if active and not previous:runs+=1
            if active and previous:adjacency+=1
            if active:possible+=1
            previous=active
    trace=sel['selection'].get('trace',[]) if sel else []
    transitions=[t.get('parent') for t in trace if t.get('parent') is not None]
    switches=sum(a!=b for a,b in zip(transitions,transitions[1:])) if transitions else None
    return dict(parents=parents,covered=covered,full=full,contiguous_runs=runs,
        adjacency_rate=adjacency/possible if possible else None,
        source_switches_selection_order=switches,trace_available=bool(sel and 'trace' in sel['selection']),
        sentences_per_parent=len(covered)/len(parents) if parents else 0.)

def load_inputs():
    # The source package is imported unchanged; its guards verify the frozen science.
    from src.post_w12.bea_v2_posthoc.protocol import context,keys,Store
    from src.post_w12.bea_v2.protocol import legacy,old as historical
    from src.post_w12.bea_v2.data import problem
    from src.post_w12.bea_v2.stages import tokenizer
    from src.post_w12.bea_qbgf.resources import load_val
    c,old,parent,v1,s,cohort,f=context()
    source=ROOT/c['output_root'];complete=read(source/'COMPLETE.json')
    if complete['logical_rows']!=1200 or complete['identity_hash']!=f['identity_hash']:
        raise ValueError('Finish the existing post-hoc 1,200-row study first; this notebook never runs missing experiments')
    store=Store(source/'rows',f['identity']);st=Store(source/'selections',f['identity'])
    if complete['keys_hash']!=digest(keys(cohort)):
        raise ValueError('Post-hoc completion has incompatible cohort/method/budget keys')
    if {p.name for p in store.path.glob('*.json')}!={store.file(k).name for k in keys(cohort)}:
        raise ValueError('Missing or unexpected source result shards; no generation fallback')
    metadata={d:{m['query_id']:m for m in cohort['selected'][d]} for d in DATASETS}
    examples={(e['dataset'],e['query_id']):e for e in load_val(parent,s,cohort)}
    tok=tokenizer(parent) # Local tokenizer only. No Qwen weights or GPU.
    bundles={};input_files={str((source/'COMPLETE.json').relative_to(ROOT)):sha(source/'COMPLETE.json')}
    for d,members in cohort['selected'].items():
        for member in members:
            q=member['query_id'];p=problem(old,v1,d,q);e=examples[d,q]
            if p['source_hash']!=digest(e['inference']) or e['cluster_id']!=member['group']:
                raise ValueError('Frozen question/source/group identity mismatch')
            pp=historical(old)/'cache/problems'/(digest([d,q])+'.json')
            input_files[str(pp.relative_to(ROOT))]=sha(pp)
            for b in [256,1024]:
                for m in METHODS:
                    key=[d,q,m,b];r=store.get(key)
                    if r is None:raise ValueError('Missing saved result; no generation fallback: '+str(key))
                    if ((r['dataset'],r['query_id'],r['method'],r['budget'])!=tuple(key) or r['cluster_id']!=member['group']
                        or r['source_hash']!=p['source_hash'] or r['evidence_tokens']>b
                        or r['reader']!='BASE_FP16' or r['model_hash']!=digest([parent['models']['Qwen'],'BASE_FP16',None])):
                        raise ValueError('Saved paired membership changed')
                    rp=store.file(key);input_files[str(rp.relative_to(ROOT))]=sha(rp)
                    item=st.get(key) if m=='BEA_V2' else None
                    if m=='BEA_V2' and item is not None:
                        input_files[str(st.file(key).relative_to(ROOT))]=sha(st.file(key))
                    if m!='BEA_V2':
                        k=[d,q,'BEA_QBGF' if m=='BEA_QBGF_V1' else m,b]
                        path=historical(old)/'cache/selections'/(digest(k)+'.json')
                        if path.exists():
                            item=legacy(old,'cache/selections',k,v1['identity_hash']);input_files[str(path.relative_to(ROOT))]=sha(path)
                        else:
                            # Isolated restoration, if it occurred, saves its own trace.
                            item=st.get(key)
                            if item is not None:input_files[str(st.file(key).relative_to(ROOT))]=sha(st.file(key))
                    units=diagnostic_units(r,p,item,tok,e)
                    bundles[d,q,b,m]=dict(row=r,problem=p,example=e,selection=item,units=units,
                        structure=structure(units,p,item),gold=retention(e,units,tok))
            print('Read-only source alignment:',d,q,flush=True)
    return bundles,tok,dict(source_identity_hash=f['identity_hash'],source_complete_sha=sha(source/'COMPLETE.json'),
        input_files=input_files,protected_inputs=c['protected_inputs']),parent

def aligned(bundles,tok):
    records=[];goldrows=[]
    keys=sorted({(d,q,b) for d,q,b,m in bundles})
    for d,q,b in keys:
        first=bundles[d,q,b,METHODS[0]];e=first['example'];p=first['problem']
        poolret=retention(e,p['pool'],tok);status=retrieval_status(poolret)
        r=dict(label=LABEL,dataset=d,query_id=q,group_id=e['cluster_id'],budget=b,question=e['question'],
            gold_answer_or_label=json.dumps(e['evaluation']['gold_label'] if d=='SciFact' else e['evaluation']['gold_answer'],ensure_ascii=False),
            score_definition='correctness' if d=='SciFact' else 'answer_F1',retrieval_status=status,
            retrieval_gold_recall=poolret['gold_evidence_recall'],mapped_annotations=poolret['mapped_annotations'],
            unmapped_annotations=poolret['unmapped_annotations'])
        scores=[];sets=[];tokens=[]
        for m,prefix in zip(METHODS,PREFIXES):
            x=bundles[d,q,b,m];row=x['row'];bs=row['behavior'];s=x['structure'];ret=x['gold']
            score=float(row['correct'] if d=='SciFact' else row['f1']);scores.append(score);sets.append(s['covered']);tokens.append(row['evidence_tokens'])
            values=dict(score=score,prediction=row['prediction'],evidence_tokens=row['evidence_tokens'],
                unused_budget=row['unused_budget'],utilization=row['budget_utilization'],
                represented_parents=bs['represented_parents'],selected_units=bs['selected_units'],FULL_usage=bs['full_units'],
                threshold_stop=row['stop_reason']=='LOW_MARGINAL_VALUE',stop_reason=row['stop_reason'],
                redundancy=bs.get('mean_pairwise_redundancy'),coverage=bs.get('aspect_coverage'),
                mean_selected_P_help=bs.get('mean_selected_p_help'),contiguous_runs=s['contiguous_runs'],
                adjacent_sentence_rate=s['adjacency_rate'],source_switches_selection_order=s['source_switches_selection_order'],
                sentences_per_parent=s['sentences_per_parent'],gold_evidence_recall=ret['gold_evidence_recall'],
                gold_parent_recall=ret['gold_parent_recall'],gold_token_coverage=ret['gold_evidence_token_coverage'],
                trace_available=s['trace_available'])
            r.update({prefix+'_'+k:v for k,v in values.items()})
            goldrows.append(dict(label=LABEL,dataset=d,query_id=q,group_id=e['cluster_id'],budget=b,method=m,
                score=score,retrieval_status=status,retrieval_gold_recall=poolret['gold_evidence_recall'],
                **{k:v for k,v in ret.items() if k!='annotation_details'}))
        similarities=[len(a&z)/len(a|z) if a|z else 1. for i,a in enumerate(sets) for z in sets[i+1:]]
        tokenrange=(max(tokens)-min(tokens))/b
        r.update(score_spread=max(scores)-min(scores),max_score=max(scores),
            token_range_fraction=tokenrange,min_sentence_jaccard=min(similarities),
            allocation_difference=max(tokenrange,1-min(similarities)),
            meaningfully_different_allocation=tokenrange>=.10 or min(similarities)<=.5)
        records.append(r)
    return records,goldrows

def candidates_for_case(case,b,record,bundles):
    """Conservative observable patterns. All mechanism labels remain candidates."""
    d,q=case['dataset'],case['query_id'];out=[]
    def add(method,label,why,confidence='CANDIDATE'):
        out.append(dict(label=LABEL,case_id=case['case_id'],dataset=d,query_id=q,budget=b,method=method,
            failure_mode=label,status=confidence,rationale=why,reviewer='',source_units_or_trace_steps=''))
    for m,prefix in zip(METHODS,PREFIXES):
        x=bundles[d,q,b,m];row=x['row'];g=x['gold'];score=record[prefix+'_score']
        better=[p for p in PREFIXES if record[p+'_score']>score]
        retention_better=[p for p in better if g['gold_evidence_recall'] is not None and record[p+'_gold_evidence_recall'] is not None and record[p+'_gold_evidence_recall']>g['gold_evidence_recall']]
        if record['retrieval_status']=='RETRIEVAL_MISS':
            add(m,'RETRIEVAL_MISS','Mapped annotated spans have no overlap with the frozen retrieved pool. Necessary-evidence absence is not established beyond supplied annotations.','OBSERVED')
        if better and retention_better and record['retrieval_status']=='RETRIEVAL_SUFFICIENT':
            add(m,'SELECTION_MISS','Complete pooled gold support was retrieved; lower-scoring method retains less annotated evidence than '+', '.join(retention_better))
        if better and retention_better and row['stop_reason']=='LOW_MARGINAL_VALUE' and row['budget_utilization']<.5:
            why=f"Threshold stopping at {row['budget_utilization']:.1%} utilization co-occurs with lower score/support retention. No alternative threshold or answer was tested."
            add(m,'OVER_PRUNING',why);add(m,'THRESHOLD_MISCALIBRATION',why)
        fragmented=[p for p in better if record[prefix+'_represented_parents']>record[p+'_represented_parents'] and
            record[prefix+'_sentences_per_parent']<record[p+'_sentences_per_parent'] and
            abs(record[prefix+'_utilization']-record[p+'_utilization'])<=.10]
        if fragmented:
            why='More represented parents but fewer underlying sentences per parent at comparable token exposure than '+', '.join(fragmented)+'. This is a fragmentation/diversity pattern, not proof of harmful redundancy penalties.'
            add(m,'OVER_DIVERSIFICATION',why);add(m,'CONTEXT_FRAGMENTATION',why)
        if better and g['gold_evidence_recall']==1 and record['retrieval_status']=='RETRIEVAL_SUFFICIENT':
            add(m,'READER_REASONING_FAILURE','All pooled annotated spans are exposed but the saved answer scores below another method. Reader utilization, annotation sufficiency, answer ambiguity and scoring remain alternatives.')
        if d=='SciFact' and row['prediction'] not in ('SUPPORT','REFUTE','NOT_ENOUGH_INFO'):
            add(m,'PARSER_OR_METRIC_EFFECT','Saved parsed output is invalid under the frozen three-label parser. Inspect raw output below; parser rules are not changed.','OBSERVED')
    v2=bundles[d,q,b,'BEA_V2'];v1=bundles[d,q,b,'BEA_QBGF_V1']
    if record['V2_score']>record['V1_score']:
        if v2['structure']['full']-v1['structure']['full']:
            add('BEA_V2','FULL_CONTEXT_RECOVERY','V2 wins and exposes additional FULL parents. Recovered source text/support is shown; the FULL contribution is not isolated causally.')
        if v2['structure']['covered']-v1['structure']['covered']:
            add('BEA_V2','COMPLEMENTARY_EVIDENCE_RECOVERY','V2 wins while exposing underlying sentences absent in V1. Whether they supply a necessary bridge requires human reading of the case.')
    if not out:add('ALL','AMBIGUOUS','Saved quality/allocation differences alone do not establish a specific mechanism.')
    return out

def format_json(obj):return '<pre>'+html.escape(json.dumps(obj,indent=2,ensure_ascii=False))+'</pre>'
def text_block(text):return '<pre>'+html.escape(str(text))+'</pre>'

def dossier(case,bundles,records):
    d,q=case['dataset'],case['query_id'];budgets=[256,1024] if case['case_type']=='BUDGET_REVERSAL' else [int(case['budget'])]
    first=bundles[d,q,budgets[0],METHODS[0]];p=first['problem'];e=first['example']
    body=f"## {case['case_id']} · {d} · {case['case_type']} · {q}\n\nQualifies for requested case pattern: **{case['qualifies']}**. {case['note']}\n\n"
    body+='### Query and retrospective annotations\n\n'+text_block(e['question'])+'\n\nGold answer/label: '+text_block(e['evaluation']['gold_label'] if d=='SciFact' else e['evaluation']['gold_answer'])
    body+='\n\nGold/support annotations (diagnostic-only; missing mappings explicit):\n\n'+format_json(annotations(e))
    body+='\n\n### Frozen SFR retrieved pool\n\n'
    for i,u in enumerate(p['pool']):
        body+=f"<details><summary>Rank {i+1} · {html.escape(u['unit_id'])} · document {html.escape(u['document_id'])} · score {p['passage_scores'][i]:.5f} · {u['token_count']} tokens</summary>\n\n"+text_block(u['source_text'])+'\n\n</details>\n\n'
    for b in budgets:
        record=records[d,q,b];body+=f"### Budget {b} · {record['retrieval_status']}\n\n"
        body+='Retrieval classification is exact overlap with mapped pooled annotations; it is not a proof that the pool contains all facts needed to answer.\n\n'
        body+='<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['FIXED','QBGF','BEA_V1','BEA_V2'])+'</tr></thead><tbody><tr>'
        for m,prefix in zip(METHODS,PREFIXES):
            x=bundles[d,q,b,m];r=x['row'];bs=r['behavior']
            body+='<td><b>'+html.escape(m)+'</b><br>'+f"Score: {record[prefix+'_score']:.4f} ({record['score_definition']})<br>Tokens: {r['evidence_tokens']}/{b}; unused: {r['unused_budget']}<br>Units: {bs['selected_units']}; parents: {bs['represented_parents']}; FULL: {bs['full_units']}<br>Stop: "+html.escape(r['stop_reason'])+'<br>Answer:'+text_block(r['prediction'])+'<details><summary>Exact saved reader context</summary>'+text_block(r['evidence_text'])+'</details></td>'
        body+='</tr></tbody></table>\n\n'
        for m in METHODS:
            x=bundles[d,q,b,m];sel=x['selection']['selection'] if x['selection'] else None
            body+='#### '+m+' — exact source units, behavior and saved decisions\n\n'
            unitrows=[dict(unit_id=u['unit_id'],parent=u['passage_id'],document=u['document_id'],kind=u['unit_type'],
                start=u['start_char'],end=u['end_char'],tokens=u['token_count'],text=u['source_text']) for u in x['units']]
            body+=format_json(dict(selected_units=unitrows,behavior=x['row']['behavior'],retention=x['gold'],
                raw_saved_output=x['row'].get('raw_prediction'),generation_cap_reached=x['row'].get('generation_cap_reached')))
            if sel:
                body+='<details><summary>Saved selection order, fidelity choices, semantic terms and stopping diagnostics</summary>\n\n'+format_json({k:sel.get(k) for k in ['selection_rank_ids','choices','trace','transition_attempts','full_attempts','step_scores','marginal_distribution','stop_reason','source_positive_contributions']})+'\n\n</details>\n\n'
            else:body+='\n\n**Allocation trace unavailable.** Source units/context are reconstructed and checked from saved IDs; no allocation is rerun and no missing trace is invented.\n\n'
        v1=bundles[d,q,b,'BEA_QBGF_V1'];v2=bundles[d,q,b,'BEA_V2']
        index={u['unit_id']:u for u in p['sentences']}
        differences={}
        for m in ['FIXED_SENTENCE','QBGF_FULL','BEA_V2']:
            recovered=bundles[d,q,b,m]['structure']['covered']-v1['structure']['covered']
            differences[m+'_sentences_absent_in_V1']=[dict(unit_id=i,parent=index[i]['passage_id'],text=index[i]['source_text']) for i in sorted(recovered)]
        body+='#### What V1 discarded / V2 recovered\n\n'+format_json(differences)
        body+='\n\nThese sentences appear only in saved alternative contexts. They were not generated after V1 stopped; no counterfactual generation or causal deletion test is performed. P_help is shown only where saved selected-transition terms contain it. Unstored candidate probabilities and rejected semantic terms are unavailable; they are not refitted or recomputed.\n\n'
    return body

REVIEW_COLUMNS=['label','case_id','dataset','query_id','budget','method','failure_mode','status',
                'rationale','reviewer','source_units_or_trace_steps']

def summarize_gold(rows):
    by={(r['dataset'],r['query_id'],r['budget'],r['method']):r for r in rows};groups=collections.defaultdict(list)
    for r in rows:
        groups[r['dataset'],r['budget'],r['method'],'ALL','ALL'].append(r)
        for m in METHODS:
            if m==r['method']:continue
            other=by[r['dataset'],r['query_id'],r['budget'],m];delta=r['score']-other['score']
            outcome='win' if delta>0 else 'loss' if delta<0 else 'tie'
            groups[r['dataset'],r['budget'],r['method'],m,outcome].append(r)
    summary=[]
    for (d,b,m,other,outcome),rs in sorted(groups.items()):
        item=dict(label=LABEL,dataset=d,budget=b,method=m,comparator=other,outcome=outcome,n_queries=len(rs),
            mapped_annotations=sum(x['mapped_annotations'] for x in rs),unmapped_annotations=sum(x['unmapped_annotations'] for x in rs),
            outcome_metric='correctness' if d=='SciFact' else 'per-query F1')
        for field in ['gold_evidence_recall','gold_parent_recall','gold_evidence_token_coverage']:
            values=[x[field] for x in rs if x[field] is not None]
            item[field]=float(np.mean(values)) if values else None;item[field+'_defined_queries']=len(values)
        summary.append(item)
    return summary

def review_table(candidates,selected):
    path=OUT/'MANUAL_REVIEW.csv'
    if not path.exists():csv(path.name,candidates,REVIEW_COLUMNS)
    reviews=pd.read_csv(path,dtype=str,keep_default_na=False).to_dict('records')
    cases={r['case_id']:r for r in selected}
    for r in reviews:
        if r['label']!=LABEL:raise ValueError('Wrong qualitative review label')
        if r['failure_mode'] not in LABELS or r['status'] not in ['CANDIDATE','OBSERVED','CONFIRMED','REJECTED','AMBIGUOUS']:
            raise ValueError('Invalid review label/status in MANUAL_REVIEW.csv')
        if r['case_id'] not in cases:raise ValueError('Unknown case ID in manual review')
        case=cases[r['case_id']]
        if (r['dataset'],r['query_id'])!=(case['dataset'],case['query_id']) or r['budget'] not in (['256','1024'] if case['case_type']=='BUDGET_REVERSAL' else [str(case['budget'])]):
            raise ValueError('Manual review case/query/budget does not match saved case selection')
        if r['method'] not in METHODS+['ALL']:raise ValueError('Unknown method in manual review')
        if r['status']=='CONFIRMED' and (not r['reviewer'].strip() or not r['rationale'].strip() or not r['source_units_or_trace_steps'].strip()):
            raise ValueError('CONFIRMED review needs reviewer, source/trace references and an evidence-based rationale')
    return reviews

def failure_counts(candidates,reviews,selected):
    result=[]
    denominator=len({(r['dataset'],r['query_id'],b) for r in selected for b in
        ([256,1024] if r['case_type']=='BUDGET_REVERSAL' else [int(r['budget'])])})
    for source,rs in [('AUTOMATIC_PATTERNS',candidates),('HUMAN_REVIEW',reviews)]:
        for mode in LABELS:
            for status in sorted({r['status'] for r in rs if r['failure_mode']==mode}):
                chosen=[r for r in rs if r['failure_mode']==mode and r['status']==status]
                unique={(r['dataset'],r['query_id'],int(r['budget'])) for r in chosen}
                result.append(dict(label=LABEL,source=source,failure_mode=mode,status=status,
                    annotation_rows=len(chosen),case_slots=len({r['case_id'] for r in chosen}),
                    unique_query_budget_count=len(unique),selected_unique_query_budget_denominator=denominator,
                    interpretation='selected-case qualitative frequency; not population prevalence or causal proof'))
    return result

def cohort_diagnostics(records,bundles):
    from src.week12.statistics import metric
    output=[]
    for d in DATASETS:
        for b in [256,1024]:
            rs=[r for r in records if r['dataset']==d and r['budget']==b]
            for m,prefix in zip(METHODS,PREFIXES):
                qualityrows=[bundles[d,r['query_id'],b,m]['row'] for r in rs]
                item=dict(label=LABEL,dataset=d,budget=b,method=m,n_queries=len(rs),
                    metric='macro_f1' if d=='SciFact' else 'answer_f1',
                    quality=metric(qualityrows,'macro_f1' if d=='SciFact' else 'f1'),
                    correctness_or_mean_query_F1=float(np.mean([r[prefix+'_score'] for r in rs])),
                    threshold_stop_fraction=float(np.mean([r[prefix+'_threshold_stop'] for r in rs])),
                    scope='entire saved 50-query development cohort; observational diagnostic, not causal experiment')
                for field in ['evidence_tokens','utilization','represented_parents','selected_units',
                    'FULL_usage','redundancy','coverage','contiguous_runs','adjacent_sentence_rate',
                    'source_switches_selection_order','sentences_per_parent','gold_evidence_recall','gold_token_coverage']:
                    values=[r[prefix+'_'+field] for r in rs if r[prefix+'_'+field] is not None]
                    item[field]=float(np.mean(values)) if values else None
                    item[field+'_defined_queries']=len(values)
                output.append(item)
    return output

def interpretation(diagnostics,selected,patterns,reviews):
    by={(r['dataset'],r['budget'],r['method']):r for r in diagnostics}
    def observation(d,b):
        parts=[]
        for m in METHODS:
            r=by[d,b,m];cov=r['gold_evidence_recall']
            parts.append(f"{m}: {r['metric']}={r['quality']:.4f}, tokens={r['evidence_tokens']:.1f}, parents={r['represented_parents']:.2f}, runs={r['contiguous_runs']:.2f}, threshold stops={r['threshold_stop_fraction']:.1%}, mapped-gold recall="+(f'{cov:.3f}' if cov is not None else 'unavailable'))
        return f'**{d} @{b}:** '+'; '.join(parts)+'.'
    content='# Final retrospective interpretation\n\n'+NOTICE+'\n\n## Full saved-cohort observations\n\n'
    content+='\n\n'.join(observation(d,b) for d in DATASETS for b in [256,1024])
    content+='\n\n## Hypotheses kept separate\n\n**QASPER @256:** Compare V1 versus V2/FIXED at matched exposure: source count, sentences per parent, adjacent-sentence rate, contiguous runs, exact support retention and discarded sentences. More parents alone does not demonstrate over-diversification, and low cosine redundancy alone does not prove an excessive penalty. The required missing rejected-candidate semantic terms may not have been saved.\n\n**QASPER @1024:** Inspect threshold stopping, unused budget and annotated support absent from V1 but present in saved alternative contexts. This is a separate potential exposure mechanism, not the same claim as fragmentation @256. No alternative answer is generated after the V1 stopping point.\n\n**HotpotQA:** Inspect V2-rescue dossiers for retention of two annotated support facts, bridge entities, and newly FULL-exposed neighboring sentences. Annotation overlap and a quality improvement do not isolate multi-hop causality.\n\n**SciFact @256:** Maximum-disagreement ties prefer FIXED-only correct cases if present. Direct claim–sentence evidence should be checked against the exact predicted label and raw output.\n\n**SciFact @1024:** Maximum-disagreement ties prefer V1-only correct cases if present. Compare stopped V2 context, retained V1 support and saved P_help terms; unavailable scores on omitted candidates are not manufactured. Low predicted usefulness for a known-correct reference is a hypothesis about utility calibration, not an observed causal usefulness label.\n\n## Reviewed case evidence\n\n'
    confirmed=[r for r in reviews if r['status']=='CONFIRMED']
    if confirmed:
        for r in confirmed:
            content+=f"- {r['case_id']} {r['dataset']} @{r['budget']} {r['method']}: **{r['failure_mode']}**, reviewed by {r['reviewer']}. {r['rationale']} References: {r['source_units_or_trace_steps']}\n"
    else:content+='No human-confirmed mechanism labels yet. Automatic patterns are candidate explanations, except explicit source-overlap/invalid-format observations. Complete MANUAL_REVIEW.csv after inspecting dossiers, then rerun this notebook to refresh the interpretation.\n'
    content+='\n## Ten final questions\n\n'
    answers=[
        ('What does FIXED do well?','It provides a simple query-ranked sentence policy. Identify its positive cells and cases in the quality/support tables; do not generalize its success across all datasets or budgets.'),
        ('What does QBGF add and where does it hurt?','Its saved per-parent fidelity choices show where whole-source context or multi-sentence states differ from FIXED. Quality differences and dilution/retention patterns are descriptive; fusion was not re-optimized here.'),
        ('What specifically caused BEA-V1’s major failures?','Causal attribution is not established by this analysis. Separate matched-exposure fragmentation @256 from threshold/exposure @1024 using exact case text and reviewed labels; retrieval misses remain upstream failures.'),
        ('What did P_help change in BEA-V2?','Inspect recovered underlying sentences, FULL transitions and saved selected P_help/IG/marginal values. The V1–V2 comparison also changes allocation/stopping, so a P_help-only causal contribution is not isolated.'),
        ('When does more context help?','Cases with better quality and recovered annotated/coherent context support a context-recovery association. They do not show that every added token helped; no new counterfactual/deletion generation is run.'),
        ('When does pruning help?','Inspect cases where a shorter saved context matches or exceeds a longer one, including near ties. This supports outcome preservation in those cases, not a universal shorter-context rule.'),
        ('When does redundancy reduction become harmful?','Only treat this as supported interpretation if exact discarded facts, saved semantic/penalty terms and reader outcomes jointly support it. Aggregate lower redundancy with lower quality is insufficient; most such flags remain hypotheses.'),
        ('Why is there no universal best allocator?','Compare task metrics separately across the six cells, strict budget reversals and case texts. Evidence needs, annotation sufficiency, packing and reader behavior can differ; mixed rankings do not identify a single cause.'),
        ('Which explanations are strongly supported by individual cases?','The reviewed-evidence list above records human-confirmed case interpretations with source/trace references. Exact context/source membership, measured quality and saved stopping events are directly observed. Human confirmation still does not turn an observational case into a causal intervention.'),
        ('Which explanations remain hypotheses?','Unreviewed mechanism candidates, excessive redundancy penalties without rejected-candidate terms, P_help miscalibration on unscored discarded candidates, and claims that FULL or multi-hop recovery alone caused an answer change remain hypotheses.')]
    for i,(q,a) in enumerate(answers,1):content+=f'{i}. **{q}** {a}\n\n'
    return content

def analyze(*,explicit=False):
    if not explicit:raise ValueError('Set RUN_ANALYSIS=True explicitly; no implicit diagnostic run')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another case-study analysis is active')
        bundles,tok,source,parent=load_inputs()
        identity=dict(label=LABEL,notice=NOTICE,source=source,selection_rules=RULES,
            implementation={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            generation_allowed=False,allocation_allowed=False,model_training_allowed=False,
            gold_used_for_retrospective_diagnosis_only=True)
        save(OUT/'ANALYSIS_IDENTITY.json',identity,immutable=True)
        save(OUT/'CASE_SELECTION_RULES.json',dict(label=LABEL,rules=RULES),immutable=True)
        records,goldrows=aligned(bundles,tok);selected,reversals=choose(records,DATASETS)
        lookup={(r['dataset'],r['query_id'],r['budget']):r for r in records}
        patterns=[]
        for case in selected:
            budgets=[256,1024] if case['case_type']=='BUDGET_REVERSAL' else [int(case['budget'])]
            for b in budgets:patterns+=candidates_for_case(case,b,lookup[case['dataset'],case['query_id'],b],bundles)
        review=review_table(patterns,selected)
        diagnostics=cohort_diagnostics(records,bundles)
        S=csv('ALIGNED_PER_QUERY.csv',records);C=csv('CASE_SELECTION.csv',[dict(r,label=LABEL) for r in selected])
        B=csv('BUDGET_REVERSALS.csv',[dict(r,label=LABEL) for r in reversals])
        G=csv('GOLD_EVIDENCE_RETENTION.csv',goldrows)
        GS=csv('GOLD_EVIDENCE_RETENTION_SUMMARY.csv',summarize_gold(goldrows))
        F=csv('FAILURE_MODE_SUMMARY.csv',failure_counts(patterns,review,selected))
        csv('AUTOMATIC_PATTERN_CANDIDATES.csv',patterns,REVIEW_COLUMNS)
        H=csv('HYPOTHESIS_DIAGNOSTICS.csv',diagnostics)
        body='# Saved-result qualitative case studies\n\n'+NOTICE+'\n\n'+C.to_markdown(index=False)
        body+='\n\nCase ranking criteria are fixed in CASE_SELECTION_RULES.json. Nonqualifying fallbacks and repeated query/budget examples are disclosed. Budget-reversal dossiers show two four-column panels. All exact contexts/answers are saved outputs, never regenerated. Missing traces/probabilities are marked unavailable.\n\n'
        for case in selected:
            body+=dossier(case,bundles,lookup)+'\n\n'
            case_patterns=[r for r in patterns if r['case_id']==case['case_id']]
            body+='### Observations and candidate mechanisms for this case\n\n'+pd.DataFrame(case_patterns)[['budget','method','failure_mode','status','rationale']].to_markdown(index=False)+'\n\n'
            case_reviews=[r for r in review if r['case_id']==case['case_id'] and r['status'] in ('CONFIRMED','REJECTED','AMBIGUOUS')]
            if case_reviews:body+='### Human review\n\n'+pd.DataFrame(case_reviews)[['budget','method','failure_mode','status','rationale','reviewer','source_units_or_trace_steps']].to_markdown(index=False)+'\n\n'
        save(OUT/'CASE_STUDIES.md',body)
        save(OUT/'FINAL_INTERPRETATION.md',interpretation(diagnostics,selected,patterns,review))
        save(OUT/'METHODOLOGY.md','# Retrospective diagnostic definitions\n\n'+NOTICE+'\n\nGold evidence recall = fraction of mapped, deduplicated annotations whose non-whitespace source content is fully covered by selected units (best alternative occurrence for repeated QASPER text). Gold parent recall = fraction of annotated native documents represented, not coarse-chunk necessity. Gold token coverage = fraction of non-whitespace tokenizer-offset gold tokens whose content is fully covered; whitespace-only gaps between exact selected source sentences are ignored, not counted as lost evidence. Unavailable if the pinned local tokenizer cannot provide offsets. Annotation pooling is not a minimal necessary-evidence set. Unmapped annotations are excluded from computable fractions and reported separately; absent NEI support produces unavailable recall, not 0 or 1. Retrieval classifications use the same annotations, and unavailable/incomplete mappings are separate statuses.\n\nFragmentation diagnostics: contiguous selected runs within each coarse parent, adjacent selected-sentence pairs / exposed underlying sentences, underlying sentences / represented parent, and source switches in saved transition order where available. FULL expands to all underlying source sentences for diagnostics. These are operational proxies, not semantic coherence judgments.\n\nAutomatic mechanism labels remain CANDIDATE. OBSERVED only denotes direct mapped-source overlap or invalid parsed output. No redundancy-overpenalization, context-dilution or P_help-miscalibration label is inferred automatically from aggregate correlation. Reviewers may add supported labels, with exact source/trace references and rationale. MANUAL_REVIEW.csv is never overwritten. Counts report selected-case slots and unique query/budget units; they are not population estimates. All paired outcome retention strata use per-query QA F1 or SciFact correctness; aggregate SciFact quality remains macro-F1.\n\nNo GPU, model loading, inference, allocation, embedding, tuning or answer regeneration occurs. A tokenizer is loaded locally on CPU for exact saved-context verification and token offsets. The notebook never launches missing study stages.\n')
        for name,h in source['input_files'].items():
            if sha(ROOT/name)!=h:raise ValueError('Read-only source changed during analysis: '+name)
        for name,h in source['protected_inputs'].items():
            if sha(ROOT/name)!=h:raise ValueError('Protected source changed during analysis: '+name)
        save(OUT/'ANALYSIS_COMPLETE.json',dict(label=LABEL,identity_hash=digest(identity),aligned_rows=len(records),
            saved_results=len(bundles),case_slots=len(selected),reader_generations=0,allocations=0,
            manual_review_sha=sha(OUT/'MANUAL_REVIEW.csv'),official_calibration='CALIBRATION_FAILED_TERMINATED'))
        return dict(aligned=S,case_selection=C,budget_reversals=B,gold_retention=G,
            gold_retention_summary=GS,failure_modes=F,hypothesis_diagnostics=H,output_root=OUT)
