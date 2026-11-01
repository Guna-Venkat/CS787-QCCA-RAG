import itertools
from .cache import digest


def action(cfg,**changes):
    changes.pop('action_id',None)
    allowed={'retriever','reader','granularity','selector','budget','compressor','representation','strategy','pool_width'}
    if set(changes)-allowed:raise ValueError('Unknown action fields; use input configuration keys: '+str(sorted(set(changes)-allowed)))
    value=dict(retriever='SFR',reader=cfg['primary_reader'],granularity='PASSAGE',selector='semantic',
        budget=512,compressor='C0',representation='P0',strategy=None,pool_width=cfg['retrieval_width'])
    value.update(changes);value['action_id']=digest(value)[:20]
    return value


def plan(cfg,phase,dataset):
    budgets=cfg['budgets'];selectors=cfg['selectors'];result=[]
    if phase=='A':
        return [action(cfg,retriever=r,granularity=g,reader=None) for r,g in itertools.product(cfg['retrievers'],cfg['retrieval_granularities'])]
    if phase=='B':return [action(cfg,representation=p,reader=None) for p in ['P0','P1','P2','P3']]
    if phase in ('C','D'):
        readers=[cfg['primary_reader']] if phase=='C' else cfg['readers']
        for reader in readers:
            result.append(action(cfg,reader=reader,granularity='NONE',selector='none',budget=0))
            result.extend(action(cfg,reader=reader,granularity=g,selector=s,budget=b)
                for g,s,b in itertools.product(cfg['retrieval_granularities'],selectors,budgets))
            result.extend(action(cfg,reader=reader,granularity='TOKEN',selector='not_applicable',budget=b,compressor='C1') for b in budgets)
    if phase=='E':
        for reader in cfg['readers']:
            result.append(action(cfg,reader=reader,granularity='NONE',selector='none',budget=0,pool_width=10))
            for comp,g in [('C0','PASSAGE'),('C1','TOKEN'),('C2','SENTENCE')]:
                result.extend(action(cfg,reader=reader,granularity=g,selector='semantic',budget=b,compressor=comp,pool_width=10) for b in budgets)
        if dataset=='QASPER':
            result.append(action(cfg,reader='Mistral',compressor='C3',budget='FULL',pool_width=10))
            result.append(action(cfg,reader='Mistral',compressor='C3',granularity='NONE',selector='none',budget=0,pool_width=10))
    if phase=='F':
        for template in cfg['finalist_run_ids']:
            if template not in cfg['finalist_configurations']:raise ValueError('Unknown manually supplied finalist ID')
            result.extend(action(cfg,**dict(cfg['finalist_configurations'][template],retriever=r)) for r in cfg['retrievers'])
    if phase=='G':
        result.append(action(cfg,granularity='NONE',selector='none',budget=0))
        result.extend(action(cfg,granularity='HIERARCHICAL',strategy=h,budget=b)
                      for h,b in itertools.product(cfg['hierarchical_strategies'],budgets))
    return result


def shared_action(a):
    # Reader-independent identity: native token counts from ALL readers are frozen.
    return {k:v for k,v in a.items() if k not in ('reader','action_id')}


def pending_reason(a,cfg):
    if (a['compressor']=='C1' or a['strategy']=='H4') and not cfg['token_compressor'].get('qualified'):
        return 'LLMLingua-2'
    return None
