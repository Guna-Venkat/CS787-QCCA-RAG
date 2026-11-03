"""Canonical core selections plus explicitly labelled demo representation policies."""
from .protocol import ROOT,OUT,DATASETS,RETRIEVERS,METHODS,BUDGETS,digest,store,grid

def choose(pool,sent,props,ps,ss,tok,budget,method):
    from src.week12.algorithm import choose as canonical,serialize,order,options
    if method not in ('PROPOSITION','MULTI_SENTENCE_2','MULTI_SENTENCE_4'):
        return canonical(pool,sent,props,ps,ss,tok,budget,method)
    if method=='PROPOSITION':
        ranked=sorted(props,key=lambda u:(-ss[u['unit_id']],order(u)));groups=[[u] for u in ranked]
    else:
        k=int(method.rsplit('_',1)[1]);groups=[]
        for p in pool:
            children=sorted([u for u in sent if u['passage_id']==p['unit_id']],key=lambda u:(-ss[u['unit_id']],order(u)))[:k]
            if children:groups.append(children)
    units=[];ranking=[]
    for g in groups:
        trial=serialize(units+g,tok)
        if trial['evidence_tokens']<=budget:units+=g;ranking.extend(u['unit_id'] for u in g)
    out=serialize(units,tok)
    out.update(method=method,budget=budget,selected_unit_ids=[u['unit_id'] for u in out['units']],selection_rank_ids=ranking,choices=[],unused_budget=budget-out['evidence_tokens'],evidence_hash=digest([out['evidence_ids'],out['text']]),source_support=1.,demo_only_representation_policy=True)
    return out

class CuratedLab:
    def __init__(self,c,tok,retriever):
        from src.week12.laboratory import Lab
        self.lab=Lab(c,tok);self.c=c;self.retriever=retriever;self.extra=None
    def prepare(self,e):
        if self.retriever!='MiniLM':return self.lab.prepare(e,self.retriever)
        # Same hierarchy and ranking policy, exact canonical MiniLM encoder / window pooling.
        from src.week10.root_cause.encoding import Encoder
        from src.week12.algorithm import cosine
        if self.extra is None:self.extra=Encoder(self.c['models']['MiniLM'],OUT/'cache/embeddings/MiniLM','cpu')
        units=self.lab.units(e);ps=[u for u in units if u['unit_type']=='PASSAGE']
        q,qcost=self.extra.encode([e['inference']['question']],'query');v,pcost=self.extra.encode([u['source_text'] for u in ps]);scores=cosine(q[0],v)
        ranked=sorted(zip(ps,scores),key=lambda t:(-t[1],tuple(t[0]['document_order']),t[0]['unit_id']))[:10]
        pool=[p for p,_ in ranked];ids={p['unit_id'] for p in pool};ss=[u for u in units if u['unit_type']=='SENTENCE' and u['passage_id'] in ids];props=[u for u in units if u['unit_type']=='PROPOSITION' and u['passage_id'] in ids]
        fine=ss+props;v,fcost=self.extra.encode([u['source_text'] for u in fine]);scores=cosine(q[0],v) if fine else []
        lineage=dict(retriever='MiniLM',encoder=self.c['models']['MiniLM'],source_hash=digest(e['inference']),candidate_ids=[p['unit_id'] for p in pool],candidate_content_hashes=[digest(p['source_text']) for p in pool],embedding_lookup_compute_seconds=sum(x['embedding_lookup_compute_seconds'] for x in (qcost,pcost,fcost)),embedding_original_compute_seconds=sum(x['embedding_original_compute_seconds'] for x in (qcost,pcost,fcost)))
        return pool,ss,props,[s for _,s in ranked],dict(zip([u['unit_id'] for u in fine],scores)),lineage
    def close(self):
        self.lab.close()
        if self.extra:self.extra.close()

def run(c,m):
    from transformers import AutoTokenizer
    from src.week12.algorithm import validate_spans,serialize
    from .data import examples
    xs=examples(c,m);st=store(m,'prepared');ps=store(m,'pools');tok=AutoTokenizer.from_pretrained(c['models']['Qwen']['id'],revision=c['models']['Qwen']['revision'],local_files_only=True)
    for rt in RETRIEVERS:
        lab=None
        try:
            for (d,q),e in xs.items():
                keys=[[d,q,rt,b,a] for b in BUDGETS for a in METHODS]
                if all(st.get(k) is not None for k in keys) and ps.get([d,q,rt]) is not None:continue
                if lab is None:lab=CuratedLab(c,tok,rt)
                pool,sent,props,scores,fine,lineage=lab.prepare(e)
                hierarchy=[dict(u,serialized_block_tokens=serialize([u],tok)['evidence_tokens']) for u in pool+sent+props]
                poolrows=[dict(p,rank=i+1,score=float(scores[i]),retriever=rt,gold_overlap=None) for i,p in enumerate(hierarchy[:len(pool)])]
                ps.put([d,q,rt],dict(dataset=d,query_id=q,retriever=rt,question=e['question'],gold=e['evaluation'],task=e['task_type'],hierarchy=hierarchy,pool=poolrows,lineage=lineage))
                for key in keys:
                    if st.get(key) is not None:continue
                    b,a=key[3:];sel=choose(pool,sent,props,scores,fine,tok,b,a);validate_spans(sel['units'],e['inference']['documents'])
                    relevance={p['unit_id']:float(s) for p,s in zip(pool,scores)};relevance.update(fine)
                    st.put(key,dict(dataset=d,query_id=q,retriever=rt,method=a,budget=b,selection=sel,lineage=lineage,relevance={u['unit_id']:relevance[u['unit_id']] for u in sel['units']}))
                print(f'Prepared curated demo {d}/{q}/{rt}; all {len(keys)} method/budget combinations durable',flush=True)
        finally:
            if lab:lab.close()
