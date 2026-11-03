"""Read-only matched slices for retrieval, reader and attribution storytelling."""
from collections import defaultdict
from statistics import mean
from demo.data_loader import load,DataError
from demo.constants import DATASETS
RETRIEVERS=('BM25','MiniLM','SFR')
READERS=('Mistral','Qwen','Llama')

TASKS={
 'QASPER':('Scientific-paper question answering','Long matching papers; scientific details may be spread across sections. No fresh final confirmation.'),
 'HotpotQA':('Multi-hop question answering','Combining evidence can matter. Search is within the provided distractor contexts, not all of Wikipedia.'),
 'SciFact':('Scientific claim verification','Predict SUPPORT, REFUTE or NOT_ENOUGH_INFO using the supplied abstract corpus. Quality is class macro-F1; example scores are correctness.')}

def model_registry():
    spec=load('w10_protocol')['models']
    return {'BM25':dict(id='Lexical BM25',revision=None),**{r:spec[r] for r in ('MiniLM','SFR')}}

def retrieval_slice(dataset,granularity,k):
    if dataset not in DATASETS or granularity not in ('PASSAGE','SENTENCE','PROPOSITION'):raise DataError('Unknown retrieval condition')
    rs=[r for r in load('retrieval_pilot') if r['dataset']==dataset and r['evidence_granularity']==granularity and r['k']==k and r['passage_representation']=='P0']
    if len(rs)!=3 or {r['retriever'] for r in rs}!=set(RETRIEVERS):raise DataError('Missing/duplicate matched retrieval slice')
    return rs

def reader_slice(dataset,budget):
    rs=[r for r in load('reader_pilot') if r['dataset']==dataset and r['budget']==budget and r['evidence_granularity'] in ('PASSAGE','SENTENCE') and r['selector']=='semantic' and r['compressor']=='C0' and r['passage_representation']=='P0' and r['strategy']=='FIXED' and r['retriever']=='SFR' and r['reader_variant']=='PUBLIC_TEXT_READER']
    keys={(r['reader'],r['evidence_granularity']) for r in rs}
    if keys!={(reader,g) for reader in READERS for g in ('PASSAGE','SENTENCE')} or len(rs)!=6:raise DataError('Missing/duplicate matched reader slice')
    return [dict(r,quality=r['macro_f1'] if dataset=='SciFact' else r['f1']) for r in rs]

def matched_retriever_downstream(dataset,budget):
    names={'BM25_PLAIN_RAG':'BM25','SFR_PLAIN_RAG':'SFR'}
    rs=[dict(r,retriever=names[r['method']]) for r in load('extension') if r['dataset']==dataset and r['budget']==budget and r['method'] in names and r['reader']=='BASE_FP16']
    if len(rs)!=2 or len({r['retriever'] for r in rs})!=2 or len({r['n_queries'] for r in rs})!=1:raise DataError('Matched extension condition unavailable')
    return rs

def sara_case():
    grouped=defaultdict(list)
    for r in load('specialists'):grouped[r['method']].append(r)
    names=('SARA_CANONICAL_BM25_K5_N10','SARA_SFR_K5_N10')
    if any(n not in grouped for n in names):raise DataError('Specialist pair missing')
    a,b=(grouped[n] for n in names)
    if len(a)!=len(b) or {r['query_id'] for r in a}!={r['query_id'] for r in b}:raise DataError('Unpaired specialist cohort')
    checkpoints={r['checkpoint_hash'] for r in a+b}
    if len(checkpoints)!=1 or any(r['reader']!='CANONICAL_ADAPTED_MISTRAL' or r['k']!=5 or r['n']!=10 or r['dataset']!='QASPER' for r in a+b):raise DataError('Incompatible SARA identities')
    return [dict(retriever=label,quality=mean(r['f1'] for r in grouped[n]),n_queries=len(grouped[n]),exact_match=mean(r['exact_match'] for r in grouped[n]),prompt_tokens=mean(r['prompt_tokens'] for r in grouped[n]),checkpoint_hash=next(iter(checkpoints))) for n,label in zip(names,('BM25','SFR'))]

def recovered_pool(query):
    hierarchy={u['unit_id']:u for u in query['hierarchy']};known={}
    for r in load('units'):
        if r['dataset']!=query['dataset'] or r['query_id']!=query['query_id']:continue
        unit=hierarchy.get(r['unit_id'])
        if not unit:continue
        parent=hierarchy.get(unit['passage_id'])
        if not parent:continue
        retriever='BM25' if r['method']=='BM25_PLAIN_RAG' else 'SFR'
        key=(retriever,parent['unit_id']);rank=r['parent_retrieval_rank'];score=r['parent_retrieval_score']
        old=known.get(key)
        if old and (old['rank']!=rank or abs(old['score']-score)>1e-8):raise DataError('Saved parent ranks/scores disagree')
        overlap=True if r.get('gold_overlap') is True else False if r['granularity']=='PASSAGE' and r.get('gold_overlap') is False else None
        if old and old['gold_overlap'] is True:overlap=True
        known[key]=dict(parent,rank=rank,score=score,retriever=retriever,gold_overlap=overlap)
    for rt in ('SFR','BM25'):
        ranks=[r['rank'] for (m,_),r in known.items() if m==rt]
        if len(set(ranks))!=len(ranks) or any(not 1<=x<=10 for x in ranks):raise DataError('Invalid recovered retrieval rank')
    return sorted(known.values(),key=lambda p:(0 if p['retriever']=='SFR' else 1,p['rank']))

def trace_comparison(query):
    pool=recovered_pool(query);groups={r:[p for p in pool if p['retriever']==r] for r in RETRIEVERS}
    for rt,rs in groups.items():
        for p in rs:
            others=[r for r in RETRIEVERS if r!=rt and any(x['unit_id']==p['unit_id'] for x in groups[r])]
            p['shared_with']=others
    return groups

def retriever_scope(query,retriever):
    if retriever not in RETRIEVERS:raise DataError('Unknown retriever')
    return dict(query,profiles=[r for r in query['profiles'] if ('BM25' if r['method']=='BM25_PLAIN_RAG' else 'SFR')==retriever])

def packing_facts(parent,budget):
    cost=parent.get('serialized_block_tokens')
    if cost is None:return dict(available=False,cost=None,budget=budget,fits=None)
    return dict(available=True,cost=cost,budget=budget,fits=cost<=budget)
