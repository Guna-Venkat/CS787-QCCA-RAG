"""No confirmation example or label is read before the explicit full-run gate."""
from .freeze_guard import require,verify,COUNTS

def load_cohort(cfg,permit):
    require(permit);f,s=verify(cfg)
    from src.week12.confirmation import examples
    xs=examples(cfg) # existing exact frozen loader; no QASPER examples when CONFIRM is empty
    expected={(d,r['query_id']):r['group'] for d,p in s['datasets'].items() for r in p['CONFIRM']}
    if len(xs)!=250 or {(e['dataset'],e['query_id']) for e in xs}!=set(expected):raise ValueError('Missing or unexpected confirmation IDs; never substitute examples')
    for e in xs:
        if e['dataset']=='QASPER' or e['cluster_id']!=expected[e['dataset'],e['query_id']]:raise ValueError('Invalid confirmation group or QASPER access')
    return xs
