"""Frozen deterministic strategies; no fitted controller or outcome inputs."""
from .selectors import rank_units


def candidate_order(question,pool,units,strategy,encoder,cfg):
    passages={p['unit_id']:p for p in pool};rank={p['unit_id']:i for i,p in enumerate(pool)}
    descendants=[u for u in units if u['passage_id'] in passages]
    sentences=[u for u in descendants if u['unit_type']=='SENTENCE']
    props=[u for u in descendants if u['unit_type']=='PROPOSITION']
    head=cfg['hierarchical']['full_head_passages'];middle=cfg['hierarchical']['sentence_passages']
    if strategy=='H0':return pool
    if strategy in ('H1','H2','H4'):
        kind=sentences if strategy in ('H1','H4') else props
        tail=[u for u in kind if rank[u['passage_id']]>=head]
        tail,_=rank_units(question,tail,'semantic',encoder,cfg['density_gamma'])
        return pool[:head]+tail
    if strategy=='H3':
        s=[u for u in sentences if rank[u['passage_id']]<middle]
        p=[u for u in props if rank[u['passage_id']]>=middle]
        return rank_units(question,s,'semantic',encoder)[0]+rank_units(question,p,'semantic',encoder)[0]
    if strategy=='H5':
        return rank_units(question,pool[:head]+sentences+props,'density',encoder,cfg['density_gamma'])[0]
    raise ValueError('Unknown hierarchy strategy')
