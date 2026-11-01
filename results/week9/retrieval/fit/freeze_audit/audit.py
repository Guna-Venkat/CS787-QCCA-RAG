"""CPU-only FIT audit from existing immutable W9-A caches; no model loading."""
import json
from pathlib import Path
import sys
import sqlite3
from contextlib import closing
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from src.week9.provenance import ROOT,branch_guard,digest,file_hash,verify_sources,source_hashes
from src.week9.config import load_config,load_frozen
from src.week9.caching import atomic_json
from src.week9.retrieval_cache import decode,read_rows
from src.week9.data import load_records
from src.week9.runner import build_identity,interventions
from src.week9.evaluation import choose_global
from src.week9.manifests import freeze

branch_guard();folder=Path(__file__).parent;base=folder.parent
cfg=load_config(ROOT/'configs/week9/w9a.yaml');cfg['_seed']=42
records,paths=load_records(cfg,'fit')
manifest=json.loads((base/'w9a_manifest.json').read_text())
assert manifest['complete'] and manifest['scope']=='full' and manifest['split']=='fit'
verify_sources(manifest['source_hashes']);assert digest(cfg)==manifest['config_hash']
identity=build_identity(cfg,'a','fit',42,paths,{})
result_path=base/'w9a_retrieval_results_compact.sqlite3'
assert json.loads(result_path.with_suffix('.identity.json').read_text())==identity
qpid={r['inference']['question_id']:r['inference']['paper_id'] for r in records}
weights=Counter(qpid.values());nq=len(records)
with closing(sqlite3.connect(result_path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
    counts=db.execute('SELECT representation,retriever,retrieval_width,count(*),count(distinct question_id) FROM entries GROUP BY representation,retriever,retrieval_width').fetchall()
    assert len(counts)==171 and all(n==nq and unique==nq for _,_,_,n,unique in counts)
    groups=db.execute('SELECT question_id,paper_id,count(*) FROM entries GROUP BY question_id,paper_id').fetchall()
    assert len(groups)==nq and all(qpid[q]==p and n==171 for q,p,n in groups)
print('Verified complete 1746 x 171 FIT grid and source/input/model identities',flush=True)
summary=pd.read_csv(base/'w9a_summary.csv')
assert len(summary)==171 and not summary.duplicated(['representation','retriever','retrieval_width']).any()
shapes=[];token_arrays=defaultdict(list)
chunk_path=base/'w9a_chunk_manifest_compact.sqlite3'
assert json.loads(chunk_path.with_suffix('.identity.json').read_text())==identity
with closing(sqlite3.connect(chunk_path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
    query='SELECT e.paper_id,e.representation,e.payload FROM entries e JOIN (SELECT paper_id,representation,min(rowid) AS first_row FROM entries GROUP BY paper_id,representation) g ON e.rowid=g.first_row ORDER BY e.rowid'
    for i,(pid,rep,blob) in enumerate(db.execute(query),1):
        row=decode(blob,digest(identity));assert row['paper_id']==pid and row['representation']==rep
        lengths=np.array([c['token_length'] for c in row['chunks']],dtype=int)
        assert len(lengths)>0
        token_arrays[rep].extend(lengths.tolist())
        shapes.append(dict(paper_id=pid,representation=rep,num_queries=weights[pid],chunks_per_paper=len(lengths),
            total_chunk_tokens=int(lengths.sum()),mean_chunk_tokens=float(lengths.mean()),median_chunk_tokens=float(np.median(lengths)),
            min_chunk_tokens=int(lengths.min()),max_chunk_tokens=int(lengths.max()),fraction_chunks_at_least_1000=float((lengths>=1000).mean())))
        if i%2000==0:print('Paper/representation chunk statistics',i,flush=True)
shape=pd.DataFrame(shapes);shape.to_csv(folder/'per_paper_chunk_statistics.csv',index=False)
assert len(shape)==len(weights)*19 and shape.groupby('representation').size().nunique()==1
reps=[]
for rep,g in shape.groupby('representation'):
    lengths=np.array(token_arrays[rep]);reps.append(dict(representation=rep,papers=len(g),queries=nq,
        pooled_mean_chunk_tokens=float(lengths.mean()),pooled_median_chunk_tokens=float(np.median(lengths)),
        pooled_p90_chunk_tokens=float(np.quantile(lengths,.9)),pooled_p95_chunk_tokens=float(np.quantile(lengths,.95)),
        paper_mean_chunk_tokens=float(g.mean_chunk_tokens.mean()),paper_median_of_chunk_medians=float(g.median_chunk_tokens.median()),
        mean_chunks_per_paper=float(g.chunks_per_paper.mean()),median_chunks_per_paper=float(g.chunks_per_paper.median()),
        p10_chunks_per_paper=float(g.chunks_per_paper.quantile(.1)),p90_chunks_per_paper=float(g.chunks_per_paper.quantile(.9)),
        query_weighted_mean_available_chunks=float(np.average(g.chunks_per_paper,weights=g.num_queries)),
        fraction_papers_fewer_than_4_chunks=float((g.chunks_per_paper<4).mean()),
        fraction_queries_fewer_than_4_chunks=float(g.loc[g.chunks_per_paper<4,'num_queries'].sum()/nq),
        fraction_chunks_at_least_1000=float((lengths>=1000).mean())))
repframe=pd.DataFrame(reps);repframe.to_csv(folder/'representation_chunk_statistics.csv',index=False)
saturation=[]
for rep,g in shape.groupby('representation'):
    for width in (10,20,40):
        actual=np.minimum(g.chunks_per_paper,width)
        saturation.append(dict(representation=rep,retrieval_width=width,
            fraction_papers_entire_pool_retrieved=float((g.chunks_per_paper<=width).mean()),
            fraction_queries_entire_pool_retrieved=float(g.loc[g.chunks_per_paper<=width,'num_queries'].sum()/nq),
            expected_mean_retrieval_actual_count=float(np.average(actual,weights=g.num_queries)),
            mean_fraction_available_chunks_retrieved=float(np.average(actual/g.chunks_per_paper,weights=g.num_queries))))
sat=pd.DataFrame(saturation);sat.to_csv(folder/'saturation.csv',index=False)
expanded=summary.merge(repframe,on='representation',validate='many_to_one').merge(sat,on=['representation','retrieval_width'],validate='many_to_one')
# Three-objective global frontier: maximize recall/density and minimize retrieved tokens.
a=expanded[['all_gold_evidence_recall','retrieved_context_tokens','evidence_overlap_density']].to_numpy()
front=[];dominated_by=[]
for i,(recall,tokens,density) in enumerate(a):
    weak=(a[:,0]>=recall-1e-12)&(a[:,1]<=tokens+1e-12)&(a[:,2]>=density-1e-12)
    strict=(a[:,0]>recall+1e-12)|(a[:,1]<tokens-1e-12)|(a[:,2]>density+1e-12)
    indices=np.where(weak&strict)[0];front.append(len(indices)==0)
    dominated_by.append('; '.join(f"{expanded.iloc[j].representation}/{expanded.iloc[j].retriever}@{expanded.iloc[j].retrieval_width}" for j in indices))
expanded['pareto_3d']=front;expanded['dominated_by']=dominated_by
expanded.to_csv(folder/'all_configuration_audit.csv',index=False)
expanded[expanded.pareto_3d].to_csv(folder/'pareto_frontier.csv',index=False)
matching=[]
for index in range(9):
    sem=summary[summary.representation==f'SEM-{index}'];fixed=summary[summary.representation==f'FIXED-{index}']
    for row in sem.merge(fixed,on=['retriever','retrieval_width'],suffixes=('_sem','_fixed')).to_dict('records'):
        entry=dict(semantic_representation=f'SEM-{index}',fixed_representation=f'FIXED-{index}',retriever=row['retriever'],retrieval_width=row['retrieval_width'])
        for metric in ['all_gold_evidence_recall','retrieved_context_tokens','evidence_overlap_density']:
            entry[metric+'_sem']=row[metric+'_sem'];entry[metric+'_fixed']=row[metric+'_fixed'];entry[metric+'_difference']=row[metric+'_sem']-row[metric+'_fixed']
        matching.append(entry)
pd.DataFrame(matching).to_csv(folder/'length_matched_fixed_vs_semantic.csv',index=False)
# Re-run the existing selection rule, not a new post-hoc preference.
options=[]
for row in summary.to_dict('records'):
    options.append(dict(row,policy=f"{row['representation']}/{row['retriever']}/{row['retrieval_width']}",quality=row['all_gold_evidence_recall'],
        measured_latency=row['retrieval_query_time']+row['index_time_embedding_seconds']+row['index_time_chunking_seconds'],
        complexity=(row['representation']!='ORIG')+(row['retriever']=='RRF')))
choice=choose_global(options,cfg['retrieval_tolerance'],primary='quality',cost='retrieved_context_tokens')
settings=dict(json.loads((base/'w9a_index_settings.json').read_text()),**{k:choice[k] for k in ['representation','retriever','retrieval_width']})
atomic_json(folder/'proposed_freeze.json',dict(settings=settings,rule='Existing preregistered global .003 recall tolerance, then retrieved tokens, latency, complexity, policy',metrics=choice,selection_split='fit',freeze_performed=False))
# Verify actual selected rows and the user's provisional point against saved summary.
checks=[]
for rep,method,width in [(choice['representation'],choice['retriever'],choice['retrieval_width']),('SEM-4','RRF',20),('SEM-4','RRF',40),('ORIG','RRF',20)]:
    rows=list(read_rows(result_path,dict(representation=rep,retriever=method,retrieval_width=int(width)),metrics_only=True))
    assert len(rows)==nq and {r['question_id'] for r in rows}==set(qpid)
    original=summary[(summary.representation==rep)&(summary.retriever==method)&(summary.retrieval_width==width)].iloc[0]
    for metric in ['all_gold_evidence_recall','retrieved_context_tokens','evidence_overlap_density']:
        actual=pd.Series([r[metric] for r in rows]).mean();assert abs(actual-original[metric])<1e-10
    checks.append(dict(representation=rep,retriever=method,retrieval_width=int(width),rows=nq,summary_matches_saved_rows=True))
    print('Validated summary against saved rows:',rep,method,width,flush=True)
# B already implements the requested fixed selector/grid and both oracle definitions.
bcfg=load_config(ROOT/'configs/week9/w9b.yaml');specs=interventions('b',bcfg,{})
assert [s['evidence_k'] for s in specs]==[0,1,2,4] and all(s['selector']=='DraftSemantic' for s in specs)
atomic_json(folder/'audit_checks.json',dict(status='PASS',split='fit',queries=nq,papers=len(weights),rows=171*nq,
    source_hashes_unchanged=identity['source_hashes']==source_hashes(),validated_summary_checks=checks,
    summary_sha256=file_hash(base/'w9a_summary.csv'),chunk_cache_sha256=file_hash(chunk_path),retrieval_cache_sha256=file_hash(result_path),
    original_selection_rule_preserved=True,model_generations=0,w9b_evidence_counts=[0,1,2,4],w9b_selector='DraftSemantic',
    epsilon_oracle=.01,w9b_training='none; retrospective oracle analysis only'))
print('AUDIT COMPLETE. Proposed:',choice['policy'],flush=True)
