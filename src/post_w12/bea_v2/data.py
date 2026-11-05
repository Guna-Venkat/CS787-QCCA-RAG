"""Exact historical LOO labels plus outcome-free canonical sentence features."""
import collections,json
import numpy as np
import pandas as pd
from .protocol import *
from src.week11.cache import digest as unit_digest

ATTR=ROOT/'results/week12/attribution/v1/analysis'

def feature_rows(problem,budget):
    """Only source units, pre-reader semantic scores and fixed budget are read."""
    ss=problem['sentences'];pool=problem['pool'];scores=problem['scores'];by={p['unit_id']:p for p in pool}
    ranks={p['unit_id']:i+1 for i,p in enumerate(pool)};ps=dict(zip([p['unit_id'] for p in pool],problem['passage_scores']))
    groups=collections.defaultdict(list)
    for i,u in enumerate(ss):groups[u['passage_id']].append(i)
    pair=np.asarray(scores['pairwise']);rows=[]
    for i,u in enumerate(ss):
        pid=u['passage_id'];ix=sorted(groups[pid],key=lambda j:(ss[j]['start_char'],ss[j]['unit_id']))
        relevance_rank=sorted(ix,key=lambda j:(-scores['relevance'][j],ss[j]['start_char'],ss[j]['unit_id'])).index(i)+1
        others=[j for j in range(len(ss)) if j!=i]
        vals=[scores['relevance'][i],ps[pid],ranks[pid],relevance_rank,
              ix.index(i)/max(1,len(ix)-1),u['token_count'],by[pid]['token_count'],len(ix),budget,len(pool),
              float(pair[i,others].mean()) if others else 0.,float(pair[i,others].max()) if others else 0.]
        row=dict(zip(FEATURES,map(float,vals)))
        if not all(np.isfinite(list(row.values()))):raise ValueError('Nonfinite feature')
        rows.append(row)
    return rows

def problem(c,v1,d,q):return legacy(c,'cache/problems',[d,q],digest(v1['input_identity']))

@exclusive
def audit(c=None):
    c=c or load();parent,v1,s,cohort=dependencies(c)
    frozen=read(ATTR/'QUANTITATIVE_RESULTS_FREEZE.json');csv=ATTR/'unit_importance.csv'
    if sha(csv)!=frozen['tables']['unit_importance']:raise ValueError('Historical label CSV differs from its freeze')
    labels=pd.read_csv(csv);units={};needed=set(labels.unit_id);sources={str(csv.relative_to(ROOT)):sha(csv),str((ATTR/'QUANTITATIVE_RESULTS_FREEZE.json').relative_to(ROOT)):sha(ATTR/'QUANTITATIVE_RESULTS_FREEZE.json')}
    sources[str((old(c)/'PILOT_COHORT.json').relative_to(ROOT))]=sha(old(c)/'PILOT_COHORT.json')
    sources[str((old(c)/'BEA_QBGF_V1_FREEZE.json').relative_to(ROOT))]=sha(old(c)/'BEA_QBGF_V1_FREEZE.json')
    for path in sorted((old(c)/'cache/hierarchy').glob('*.json')):
        if path.name=='identity.json':continue
        e=read(path)
        if e['checksum']!=digest(e['row']):raise ValueError('Corrupt hierarchy cache')
        matched=False
        for u in e['row']:
            if u['unit_id'] not in needed:continue
            identity=unit_digest([u['dataset'],u['document_id'],u['unit_type'],u['start_char'],u['end_char'],u['source_text']])[:32]
            if identity!=u['unit_id'] or unit_digest(u['source_text'])!=u['hash']:raise ValueError('Exact source reconstruction failed')
            if u['unit_id'] in units and units[u['unit_id']]['source_text']!=u['source_text']:raise ValueError('Ambiguous source unit')
            units[u['unit_id']]=u;matched=True
        if matched:sources[str(path.relative_to(ROOT))]=sha(path)
    missing=needed-units.keys()
    if missing:
        text(root(c)/'AUDIT_ATTRIBUTION_DATA.md','# Attribution audit — BLOCKED\n\nMissing exact source identities: '+str(len(missing)))
        raise ValueError('Cannot map historical labels to exact source units; stopping')
    metadata={(d,r['query_id']):r for d,rs in cohort['selected'].items() for r in rs}
    problems={};features={};usable=[];excluded=collections.Counter();rows=[]
    for r in labels.to_dict('records'):
        d,q=r['dataset'],str(r['query_id']);u=units[r['unit_id']]
        if (d,q) not in metadata:raise ValueError('Attribution query is not in the frozen development cohort')
        meta=metadata[d,q]
        if r['cluster_id']!=meta['group']:raise ValueError('Attribution/frozen grouping mismatch')
        expected='HELPFUL' if r['importance']>0 else 'HARMFUL' if r['importance']<0 else 'NEUTRAL'
        if expected!=r['category'] or abs(r['full_score']-r['ablated_score']-r['importance'])>1e-9:raise ValueError('LOO sign/category mismatch')
        if u['unit_type']!=r['granularity'] or u['token_count']!=r['tokens']:raise ValueError('Unit granularity/token identity mismatch')
        rows.append(dict(dataset=d,category=r['category'],granularity=r['granularity']))
        if r['granularity']!='SENTENCE':excluded['non_sentence_labels']+=1;continue
        if r['method']=='BM25_PLAIN_RAG':excluded['BM25_context_not_SFR']+=1;continue
        if (d,q) not in problems:
            problems[d,q]=problem(c,v1,d,q)
            path=old(c)/'cache/problems'/(digest([d,q])+'.json');sources[str(path.relative_to(ROOT))]=sha(path)
        p=problems[d,q];sentence_index={u['unit_id']:i for i,u in enumerate(p['sentences'])}
        if r['unit_id'] not in sentence_index:excluded['outside_current_frozen_SFR_pool']+=1;continue
        key=(d,q,int(r['budget']))
        if key not in features:features[key]=feature_rows(p,int(r['budget']))
        i=sentence_index[r['unit_id']];f=features[key][i]
        if abs(f['query_sentence_cosine']-r['relevance'])>2e-5 or abs(f['passage_query_cosine']-r['parent_retrieval_score'])>2e-5:
            raise ValueError('Historical/current semantic scores disagree; no approximate feature join')
        if f['retrieval_rank']!=r['parent_retrieval_rank']:raise ValueError('Historical source rank differs')
        usable.append(dict(dataset=d,query_id=q,group=meta['group'],paper_id=meta['paper_id'],method=r['method'],budget=int(r['budget']),
                           unit_id=u['unit_id'],parent_id=u['passage_id'],source_text=u['source_text'],start_char=u['start_char'],end_char=u['end_char'],
                           category=r['category'],importance=r['importance'],**f))
    frame=pd.DataFrame(usable)
    if frame.empty:raise ValueError('No exact usable training labels')
    summary=frame.groupby(['dataset','category']).size().rename('units').reset_index()
    counts=frame.groupby('dataset').agg(units=('unit_id','size'),unique_units=('unit_id','nunique'),queries=('query_id','nunique'),groups=('group','nunique')).reset_index()
    schema=dict(features=FEATURES,hash=digest(FEATURES),all_pre_reader=True,query_length_omitted='Not needed; original query token lengths are not cached',
                sentence_rank='Descending query cosine within parent, source-offset tie break',position='Source-order index/(parent sentence count-1)',
                eligibility='Exact SENTENCE identity, current frozen SFR pool, non-BM25 historical context; never assign passage/clause labels to sentences',
                targets=['HELPFUL','HARMFUL'],neutral='Negative class for each binary target; not synonymous with useless')
    write(root(c)/'FEATURE_SCHEMA.json',schema)
    write(root(c)/'TRAINING_SOURCES.json',sources)
    # New V2 files only; inference uses the separate strict feature_rows allowlist.
    text(root(c)/'utility_model/training_units.csv',frame.to_csv(index=False))
    write(root(c)/'ATTRIBUTION_AUDIT.json',dict(rows=len(labels),exact_unit_identities=len(needed),usable_rows=len(frame),excluded=dict(excluded),class_counts=summary.to_dict('records'),source_hashes=sources))
    text(root(c)/'AUDIT_ATTRIBUTION_DATA.md','# Attribution-data audit\n\nAll '+str(len(labels))+' historical unit rows map to '+str(len(needed))+' exact content/offset-derived unit IDs. Original prepared/perturbation shards were deleted; labels survive in the checksum-frozen historical CSV. All label signs and original/ablated differences were verified exactly.\n\n'+counts.to_markdown(index=False)+'\n\n'+summary.to_markdown(index=False)+'\n\nExcluded observations: '+json.dumps(dict(excluded))+'.\n\nOnly exact sentence labels are trained; passage/clause outcomes are not transferred to constituents. Multiple historical contexts may label the same sentence differently; this intervention-specific uncertainty is retained. Models are dataset-specific, and all examples of the same frozen group remain in one fold. Query token length is omitted; other chosen features are present in canonical cached source/scoring metadata. Gold overlap, labels, answer scores/text and LOO magnitudes are never predictor inputs. Severe neutral-class imbalance is reported, without class balancing that would distort the probability target. Historical selected-unit sampling is not unbiased candidate sampling.\n')
    return frame,cohort
