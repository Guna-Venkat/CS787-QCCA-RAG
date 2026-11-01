"""CPU-only fabricated fixtures; never consult confirmation examples or models."""
from copy import deepcopy
from src.week12.protocol import load,digest,immutable
from src.week12.runner import fixture_examples
from src.week12.laboratory import Lab
from src.week12.reader import prompt_ids
from .freeze_guard import folder,protocol
from .confirmation_runner import _prepare,_generate
class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(map(ord,text))
    def decode(self,ids,**kw):return ''.join(map(chr,ids))
    def apply_chat_template(self,messages,**kw):return '<user>'+messages[0]['content']+'</user><assistant>'
class Reader:
    calls=0
    def __init__(self,cfg,tok):self.cfg=cfg;self.tok=tok
    def generate(self,e,s):
        Reader.calls+=1
        raw='SUPPORT' if e['task_type']=='verification' else 'synthetic answer'
        return dict(raw_prediction=raw,prediction=raw,prompt_tokens=len(prompt_ids(self.tok,e['question'],e['task_type'],s['evidence_ids'],self.cfg['labels'])),generated_tokens=len(raw),generation_seconds=0.,prompt_hash=digest(prompt_ids(self.tok,e['question'],e['task_type'],s['evidence_ids'],self.cfg['labels'])),generation_cap_reached=False,reader='BASE_FP16',model_hash=digest([self.cfg['models']['Qwen'],'BASE_FP16',None]))
    def close(self):pass

def run(out=None):
    cfg=deepcopy(load());out=out or folder(True);out.mkdir(parents=True,exist_ok=True)
    # Dedicated temporary/synthetic hierarchy cache, never a real model cache.
    cfg['cache_root']=str(out/'fixture_cache');xs=[e for e in fixture_examples() if e['dataset']!='QASPER'];tok=Tokenizer();ident=protocol(synthetic=True)
    immutable(out/'protocol.json',ident)
    _prepare(cfg,xs,tok,Lab(cfg,tok,synthetic=True),out,ident)
    _generate(cfg,xs,tok,lambda:Reader(cfg,tok),out,ident,True)
    return out,ident
