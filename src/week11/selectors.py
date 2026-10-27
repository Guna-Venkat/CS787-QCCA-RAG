import numpy as np
from .retrievers import ranked, retrieve


def normalize(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return values
    span=values.max()-values.min()
    return (values-values.min())/span if span>1e-12 else np.zeros_like(values)


def select_scores(lexical,semantic,token_counts,method,gamma=1.):
    if method=='lexical':return np.asarray(lexical)
    if method=='semantic':return np.asarray(semantic)
    if method=='hybrid':return .5*normalize(semantic)+.5*normalize(lexical)
    if method=='density':
        return np.asarray(semantic)/np.maximum(1,np.asarray(token_counts))**gamma
    raise ValueError('Unknown selector')


def rank_units(question,units,method,encoder,gamma=1.):
    lex,lexcost=retrieve(question,units,'BM25') if method in ('lexical','hybrid') else ([],{})
    if method=='lexical':return lex,lexcost
    dense,cost=retrieve(question,units,'SFR',encoder)
    l={u['unit_id']:u['score'] for u in lex};d={u['unit_id']:u['score'] for u in dense}
    scores=select_scores([l.get(u['unit_id'],0.) for u in units],[d[u['unit_id']] for u in units],
                         [u['token_count'] for u in units],method,gamma)
    return ranked(units,scores),cost
