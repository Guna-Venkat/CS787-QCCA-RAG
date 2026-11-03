"""Publish only a complete, self-contained grid. Old research exports are not modified."""
from .protocol import OUT,APP,DATASETS,RETRIEVERS,METHODS,BUDGETS,grid,store,immutable,read,sha,digest

def status(m):
    stores={name:store(m,name) for name in ('prepared','results','attribution','pools')}
    keys=grid(m);ready=answers=loo=expected_loo=0
    for k in keys:
        key=list(k);p=stores['prepared'].get(key);r=stores['results'].get(key)
        ready+=p is not None;answers+=r is not None
        if p:
            for u in p['selection']['units']:
                expected_loo+=1;loo+=stores['attribution'].get(key+[u['unit_id']]) is not None
    return dict(expected_conditions=len(keys),prepared=ready,answers=answers,expected_loo=expected_loo,completed_loo=loo,
                answers_complete=answers==len(keys),all_views_complete=answers==len(keys) and ready==len(keys) and loo==expected_loo,
                published=(OUT/'PUBLISHED.json').exists(),scope=m['scope'])

def publish(m):
    from src.week12.algorithm import order
    s=status(m)
    if not s['all_views_complete']:raise ValueError('Publish requires every answer and every selected-unit deletion: '+str(s))
    ps=store(m,'pools');pr=store(m,'prepared');rs=store(m,'results');ats=store(m,'attribution')
    queries={}
    for d in DATASETS:
        for q in m['queries'][d]:
            pools={rt:ps.get([d,q,rt]) for rt in RETRIEVERS}
            if any(v is None for v in pools.values()):raise ValueError('Missing complete source pool')
            first=pools[RETRIEVERS[0]];conditions={}
            for rt in RETRIEVERS:
                h={u['unit_id']:u for u in pools[rt]['hierarchy']}
                for b in BUDGETS:
                    for a in METHODS:
                        key=[d,q,rt,b,a];p=pr.get(key);r=rs.get(key);sel=p['selection'];units=[]
                        if r['evidence_hash']!=digest([sel['evidence_ids'],sel['text']]):raise ValueError('Answer/selected-evidence mismatch')
                        if sel['evidence_tokens']>b:raise ValueError('Budget violation')
                        for i,u in enumerate(sel['units']):
                            loo=ats.get(key+[u['unit_id']]);source=h[u['unit_id']]
                            if source['source_text']!=u['source_text']:raise ValueError('Published source identity mismatch')
                            units.append(dict(source=source,text=u['source_text'],unit_id=u['unit_id'],unit_index=i,granularity=u['unit_type'],tokens=u['token_count'],relevance=p['relevance'][u['unit_id']],**{k:v for k,v in loo.items() if k!='unit_id'}))
                        conditions[digest([rt,b,a])]=dict(result=r,selection=sel,units=units)
            queries[d+':'+q]=dict(dataset=d,query_id=q,question=first['question'],gold=first['gold'],task=first['task'],pools=pools,conditions=conditions)
    # One compact portable bundle, including sources/answers/deletions; does not depend on old caches.
    bundle=dict(manifest=m,queries=queries,coverage=dict(s,published=True),source='new_curated_demo_v1')
    path=OUT/'bundle.json';immutable(path,bundle)
    immutable(OUT/'PUBLISHED.json',dict(bundle=path.name,sha256=sha(path),manifest_hash=digest(m),conditions=len(grid(m)),all_views_complete=True))
    return dict(path=str(path),conditions=len(grid(m)),queries=len(queries))

from functools import lru_cache

@lru_cache(maxsize=2)
def _cached_bundle(receipt_path, receipt_mtime, bundle_mtime, bundle_size):
    p=__import__('pathlib').Path(receipt_path)
    receipt=read(p);path=p.parent/receipt['bundle']
    if path.resolve().parent!=p.parent.resolve() or sha(path)!=receipt['sha256']:raise ValueError('Published demo bundle checksum mismatch')
    bundle=read(path)
    if digest(bundle['manifest'])!=receipt['manifest_hash'] or receipt['conditions']!=len(grid(bundle['manifest'])):raise ValueError('Published demo registry mismatch')
    if not receipt['all_views_complete']:raise ValueError('Incomplete demo bundle cannot be presented as complete')
    return bundle


def load_bundle():
    p=OUT/'PUBLISHED.json'
    if not p.exists():return None
    receipt=read(p);path=OUT/receipt['bundle']
    if path.resolve().parent!=OUT.resolve():raise ValueError('Published bundle path escapes app data')
    st=path.stat()
    return _cached_bundle(str(p),p.stat().st_mtime_ns,st.st_mtime_ns,st.st_size)
