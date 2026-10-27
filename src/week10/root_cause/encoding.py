"""Sequential encoders and persistent per-text embeddings; no LM co-residency."""
import gc
import sqlite3
import time
import numpy as np
from .storage import digest

SFR_INSTRUCTION='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '

class Encoder:
    def __init__(self,spec,directory,device='cpu'):
        self.spec=spec; self.directory=directory; self.device=device; self.model=None
        directory.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(directory/'embeddings.sqlite3')
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB, dim INTEGER, seconds REAL)')
    def key(self,text,role):
        return digest(dict(model=self.spec,role=role,text=text,normalization=False,pooling='explicit-window-mean-v1',
                           query_prefix=SFR_INSTRUCTION if role=='query' and 'SFR-' in self.spec['id'] else ''))
    def encode(self,texts,role='document',allow_compute=True):
        result=[]; physical=0.; conceptual=0.; hits=0; started=time.perf_counter()
        for text in texts:
            key=self.key(text,role); row=self.db.execute('SELECT vector,dim,seconds FROM vectors WHERE key=?',(key,)).fetchone()
            if row is None:
                if not allow_compute: raise RuntimeError('Embedding not prepared; run retrieval/preprocessing before loading generator')
                if self.model is None:
                    from sentence_transformers import SentenceTransformer
                    if self.device=='cuda':
                        from .runtime import require_memory
                        require_memory(14.2)
                    import torch
                    opts={'torch_dtype':torch.float16} if self.device=='cuda' else {}
                    self.model=SentenceTransformer(self.spec['id'],revision=self.spec['revision'],device=self.device,
                        local_files_only=True,model_kwargs=opts)
                value=(SFR_INSTRUCTION+text) if role=='query' and 'SFR-' in self.spec['id'] else text
                tok=self.model.tokenizer; ids=tok.encode(value,add_special_tokens=False)
                width=self.model.max_seq_length-2
                # No silently truncated long inputs. Normal chunks stay in one window.
                windows=[tok.decode(ids[i:i+width],skip_special_tokens=True) for i in range(0,len(ids),width)] or ['']
                begin=time.perf_counter()
                vector=np.asarray(self.model.encode(windows,batch_size=1,normalize_embeddings=False,show_progress_bar=False),dtype=np.float32).mean(0)
                seconds=time.perf_counter()-begin; physical+=seconds
                self.db.execute('INSERT INTO vectors VALUES (?,?,?,?)',(key,vector.tobytes(),len(vector),seconds)); self.db.commit()
                row=(vector.tobytes(),len(vector),seconds)
            else: hits+=1
            vector=np.frombuffer(row[0],dtype=np.float32).copy()
            if len(vector)!=row[1] or not np.isfinite(vector).all(): raise ValueError('Corrupt embedding')
            result.append(vector); conceptual+=row[2]
        return np.asarray(result),dict(embedding_lookup_compute_seconds=time.perf_counter()-started,
            embedding_physical_compute_seconds=physical,embedding_original_compute_seconds=conceptual,embedding_cache_hits=hits)
    def close(self):
        self.model=None; self.db.close(); gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def rank(inference,method,encoder=None):
    from src.week9.retrieval import bm25
    import time
    chunks=inference['chunks']; texts=[c['text'] for c in chunks]; t=time.perf_counter(); cost={}
    if method=='BM25': scores=bm25(inference['question'],texts)
    else:
        vec,cost=encoder.encode(texts); q,qcost=encoder.encode([inference['question']],'query')
        document_cost=cost.copy()
        cost={k:cost[k]+qcost[k] for k in cost}
        cost.update(document_preprocessing_lookup_seconds=document_cost.get('embedding_lookup_compute_seconds',0.),
            query_embedding_lookup_seconds=qcost.get('embedding_lookup_compute_seconds',0.),
            query_original_embedding_seconds=qcost.get('embedding_original_compute_seconds',0.))
        scores=unit(vec)@unit(q)[0]
    ix=sorted(range(len(chunks)),key=lambda i:(-float(scores[i]),chunks[i]['order'],chunks[i]['chunk_id']))
    return [dict(chunks[i],score=float(scores[i]),rank=j+1) for j,i in enumerate(ix)],dict(cost,ranking_total_seconds=time.perf_counter()-t)
