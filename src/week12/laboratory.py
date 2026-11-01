"""New caches only; read-only reuse of pinned W11 embeddings is audited."""
from pathlib import Path
import sqlite3,time
import numpy as np
from .protocol import ROOT,Store,digest
from .algorithm import cosine
from src.week10.root_cause.encoding import Encoder as CanonicalEncoder

class Encoder(CanonicalEncoder):
    def __init__(self,cfg):
        super().__init__(cfg['models']['SFR'],ROOT/cfg['cache_root']/'embeddings','cuda');self.reused=0
        self.history=ROOT/'cache/week11/v1/embeddings/shared_units/SFR/embeddings.sqlite3'
    def encode(self,texts,role='document',allow_compute=True):
        begin=time.perf_counter();reused=0
        if self.history.exists():
            old=sqlite3.connect(f'file:{self.history}?mode=ro',uri=True)
            try:
                for text in texts:
                    key=self.key(text,role)
                    if self.db.execute('SELECT 1 FROM vectors WHERE key=?',(key,)).fetchone():continue
                    row=old.execute('SELECT vector,dim,seconds FROM vectors WHERE key=?',(key,)).fetchone()
                    if row:self.db.execute('INSERT INTO vectors VALUES (?,?,?,?)',(key,*row));reused+=1
                self.db.commit()
            finally:old.close()
        values,meta=super().encode(texts,role,allow_compute);meta.update(readonly_historical_reuse=reused,readonly_reuse_lookup_seconds=time.perf_counter()-begin);self.reused+=reused
        return values,meta

class Lab:
    def __init__(self,cfg,tok,synthetic=False):
        self.cfg=cfg;self.tok=tok;self.synthetic=synthetic;self._encoder=None;self.units_cache={};self.ranks={}
        self.disk=Store(ROOT/cfg['cache_root']/('synthetic_hierarchy' if synthetic else 'hierarchy'),dict(hierarchy=cfg['hierarchy'],tokenizer=cfg['models']['Qwen'],source_builder='readonly-W11-build'))
    @property
    def encoder(self):
        if self._encoder is None:
            if self.synthetic:
                from src.week11.embeddings import SyntheticEncoder
                self._encoder=SyntheticEncoder()
            else:self._encoder=Encoder(self.cfg)
        return self._encoder
    def units(self,example):
        from src.week11.hierarchy import build
        units=[]
        for index,d in enumerate(example['inference']['documents']):
            key=digest([d,self.cfg['hierarchy']])
            if key not in self.units_cache:
                val=self.disk.get(key)
                if val is None:val=build([d],self.tok,self.cfg['hierarchy']);self.disk.put(key,val)
                self.units_cache[key]=val
            units.extend(dict(u,document_order=[index]+u['document_order'][1:]) for u in self.units_cache[key])
        return units
    def prepare(self,example,retriever='SFR'):
        started=time.perf_counter();units=self.units(example);passages=[u for u in units if u['unit_type']=='PASSAGE']
        question=example['inference']['question'];costs=[]
        if retriever=='SFR':q,qcost=self.encoder.encode([question],'query');costs.append(qcost)
        else:q=None
        if retriever=='BM25':
            from src.week9.retrieval import bm25
            ps=list(map(float,bm25(question,[u['source_text'] for u in passages])))
        else:
            vec,cost=self.encoder.encode([u['source_text'] for u in passages]);costs.append(cost);ps=cosine(q[0],vec)
        ranked=sorted(zip(passages,ps),key=lambda x:(-x[1],tuple(x[0]['document_order']),x[0]['unit_id']))[:self.cfg['candidate_width']]
        pool=[p for p,_ in ranked];ps=[s for _,s in ranked];ids={u['unit_id'] for u in pool}
        sentences=[u for u in units if u['unit_type']=='SENTENCE' and u['passage_id'] in ids];props=[u for u in units if u['unit_type']=='PROPOSITION' and u['passage_id'] in ids]
        fine=sentences+props
        if retriever=='BM25':
            from src.week9.retrieval import bm25
            scores=list(map(float,bm25(question,[u['source_text'] for u in fine])))
        else:
            vec,cost=self.encoder.encode([u['source_text'] for u in fine]);costs.append(cost);scores=cosine(q[0],vec) if fine else []
        ss={u['unit_id']:v for u,v in zip(fine,scores)}
        return pool,sentences,props,ps,ss,dict(preparation_seconds=time.perf_counter()-started,retriever=retriever,source_hash=digest(example['inference']),candidate_ids=[p['unit_id'] for p in pool],candidate_content_hashes=[digest(p['source_text']) for p in pool],encoder=self.cfg['models']['SFR'],query_instruction_role='canonical SFR query instruction; documents unprefixed',historical_reused_vectors=getattr(self._encoder,'reused',0),embedding_original_compute_seconds=sum(c.get('embedding_original_compute_seconds') or 0 for c in costs),embedding_lookup_compute_seconds=sum(c.get('embedding_lookup_compute_seconds') or 0 for c in costs))
    def close(self):
        if self._encoder is not None:
            close=getattr(self._encoder,'close',None)
            if close:close()
            self._encoder=None
