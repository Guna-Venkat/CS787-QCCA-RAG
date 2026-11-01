import time
import numpy as np
from .embeddings import unit
from src.week9.retrieval import bm25


class BM25Index:
    def __init__(self,texts):
        import re
        from collections import Counter
        self.n=len(texts);self.lengths=[];self.postings={}
        for i,text in enumerate(texts):
            counts=Counter(re.findall(r'\w+',text.lower()));self.lengths.append(sum(counts.values()))
            for word,tf in counts.items():self.postings.setdefault(word,[]).append([i,tf])
        self.average=max(float(np.mean(self.lengths)) if self.lengths else 0.,1.)
    def score(self,query):
        import re
        scores=np.zeros(self.n)
        for word in re.findall(r'\w+',query.lower()):
            docs=self.postings.get(word,[]);idf=np.log(1+(self.n-len(docs)+.5)/(len(docs)+.5))
            for i,tf in docs:scores[i]+=idf*tf*2.5/(tf+1.5*(.25+.75*self.lengths[i]/self.average))
        return scores
    def serialize(self):return vars(self)
    @classmethod
    def restore(cls,value):
        obj=cls([]);obj.__dict__.update(value);return obj


def ranked(units,scores):
    if len(units)!=len(scores) or not np.isfinite(scores).all():raise ValueError('Invalid unit/score alignment')
    order=sorted(range(len(units)),key=lambda i:(-float(scores[i]),units[i]['document_order'],units[i]['unit_id']))
    return [dict(units[i],score=float(scores[i]),rank=j+1) for j,i in enumerate(order)]


def retrieve(question,units,method,encoder=None):
    started=time.perf_counter();cost={}
    if not isinstance(question,str):raise ValueError('Only inference query text enters retrieval')
    if not units:return [],dict(retrieval_latency=time.perf_counter()-started)
    texts=[u['source_text'] for u in units]
    if method=='BM25':scores=bm25(question,texts)
    elif method in ('MiniLM','SFR'):
        if encoder is None:raise ValueError('Dense retriever requires pinned encoder')
        docs,dc=encoder.encode(texts);q,qc=encoder.encode([question],'query')
        scores=unit(docs)@unit(q)[0]
        cost=dict(document_preprocessing=dc,query_embedding=qc)
    else:raise ValueError('Unknown retriever')
    return ranked(units,scores),dict(cost,retrieval_latency=time.perf_counter()-started)


def passage_scores(query_vector,passage_vectors,sentence_vectors,representation,beta=5.):
    q=unit([query_vector])[0];direct=unit(passage_vectors)
    if representation=='P0':return direct@q
    scores=[]
    for i,v in enumerate(sentence_vectors):
        if not len(v):scores.append(float(direct[i]@q));continue
        v=unit(v);sim=v@q
        if representation=='P1':score=unit([v.mean(0)])[0]@q
        elif representation=='P2':score=sim.max()
        elif representation=='P3':
            weights=np.exp(beta*(sim-sim.max()));weights/=weights.sum()
            score=unit([weights@v])[0]@q
        else:raise ValueError('Unknown passage representation')
        scores.append(float(score))
    return np.asarray(scores)
