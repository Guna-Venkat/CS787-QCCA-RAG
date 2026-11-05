"""Frozen encoders/reader; only new caches writable, old SQLite opened read-only."""
import copy,json,sqlite3
import numpy as np
from src.week12.protocol import ROOT,digest
from src.week12.laboratory import Lab
from src.week10.root_cause.encoding import Encoder as CanonicalEncoder
from .protocol import output,read

class Encoder(CanonicalEncoder):
    def __init__(self,parent,c):
        super().__init__(parent['models']['SFR'],output(c)/'cache/embeddings','cuda')
        self.sources=[ROOT/parent['cache_root']/'embeddings/embeddings.sqlite3',
                      ROOT/'cache/week11/v1/embeddings/shared_units/SFR/embeddings.sqlite3']
        self.reused=0
        self.legacy_rejections=[]
    def encode(self,texts,role='document',allow_compute=True):
        for p in self.sources:
            if not p.exists():continue
            old=None
            try:
                old=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)
                for text in texts:
                    key=self.key(text,role)
                    if self.db.execute('SELECT 1 FROM vectors WHERE key=?',(key,)).fetchone():continue
                    row=old.execute('SELECT vector,dim,seconds FROM vectors WHERE key=?',(key,)).fetchone()
                    if row is not None:
                        v=np.frombuffer(row[0],dtype=np.float32)
                        if len(v)!=row[1] or not np.isfinite(v).all():
                            self.legacy_rejections.append(dict(path=str(p),key=key,reason='invalid legacy vector'));continue
                        self.db.execute('INSERT INTO vectors VALUES (?,?,?,?)',(key,*row));self.reused+=1
                self.db.commit()
            except (sqlite3.Error,ValueError) as exc:self.legacy_rejections.append(dict(path=str(p),reason=type(exc).__name__+': '+str(exc)))
            finally:
                if old is not None:old.close()
        self.db.commit()
        values,meta=super().encode(texts,role,allow_compute)
        meta.update(readonly_legacy_reused=self.reused,legacy_rejections=list(self.legacy_rejections))
        return values,meta

class IsolatedLab(Lab):
    def __init__(self,parent,tok,c):
        local=copy.deepcopy(parent);local['cache_root']=str((output(c)/'cache').relative_to(ROOT))
        super().__init__(local,tok);self.original_parent=parent;self.extension=c
    @property
    def encoder(self):
        if self._encoder is None:self._encoder=Encoder(self.original_parent,self.extension)
        return self._encoder

def load_val(parent,s,cohort):
    """Same frozen native adapters, restricted to selected VAL IDs before normalizing."""
    from src.week12.splits import source_paths,records_for_validation
    from src.week12.data import native_qasper
    from src.week11.datasets import hotpot,scifact,read_jsonl
    paths=source_paths(parent);out=[]
    for d,members in cohort['selected'].items():
        want={r['query_id'] for r in members};meta={r['query_id']:r for r in s['datasets'][d]['VAL']}
        if not want<=meta.keys() or want&{r['query_id'] for r in s['datasets'][d]['CONFIRM']}:raise ValueError('Not VAL-only')
        if d=='QASPER':
            validation_papers={r['paper_id'] for r in records_for_validation(parent)}
            roles={('val' if meta[q]['group'] in validation_papers else 'train') for q in want};xs=[]
            for role in sorted(roles):xs.extend(native_qasper(paths['QASPER_'+role],allowed_ids=want))
        elif d=='HotpotQA':xs=hotpot([r for r in read(paths['HotpotQA_val']) if str(r['_id']) in want],'dev')
        elif d=='SciFact':xs=scifact([r for r in read_jsonl(paths['SciFact_val']) if str(r['id']) in want],read_jsonl(paths['SciFact_corpus']),'dev')
        else:raise ValueError('Unexpected dataset')
        if {e['query_id'] for e in xs}!=want:raise ValueError('Missing selected VAL IDs')
        for e in xs:
            e['cluster_id']=meta[e['query_id']]['group'];e['partition']='VAL';e['split']='dev';e['inference']['split']='dev'
        out.extend(xs)
    return sorted(out,key=lambda e:(e['dataset'],e['query_id']))

def legacy_get(parent,f,folder,key):
    """Never instantiate frozen Store (constructor writes identity.json)."""
    folder=ROOT/parent['output_root']/folder
    ident=folder/'identity.json';path=folder/(digest(key)+'.json')
    if not ident.exists():return None,('rejected: legacy row has no verifiable store identity' if path.exists() else 'missing')
    if not path.exists():return None,'missing'
    try:
        identity=read(ident);meta=identity.get('parent',identity)
        extra=meta.get('extra',{})
        if dict(meta,extra={})!=dict(f['identity'],extra={}):raise ValueError('parent code/config/source/model/package identity differs')
        if extra.get('synthetic') is not False or extra.get('real_smoke') is not False or extra.get('adapter_hash') is not None:
            raise ValueError('not unchanged full BASE_FP16 output')
        v=read(path)
        if v['key']!=key or v['identity']!=digest(identity) or v['checksum']!=digest(v['row']):raise ValueError('envelope/checksum mismatch')
        return v['row'],str(path.relative_to(ROOT))
    except (OSError,ValueError,KeyError,TypeError) as exc:return None,'rejected: '+str(exc)

def compatible_prediction(parent,f,e,selection,tok):
    """Validate exact current source/selection/prompt before borrowing an answer."""
    from src.week12.reader import prompt_ids,parse_label
    diagnostics=[];method=selection['method'];budget=selection['budget']
    expected_prompt=digest(prompt_ids(tok,e['question'],e['task_type'],selection['evidence_ids'],parent['labels']))
    candidates=[]
    if method in ('FIXED_SENTENCE','QBGF_FULL'):
        for phase in ('C','D'):
            row,source=legacy_get(parent,f,'validation/'+phase+'/rows',[e['dataset'],e['query_id'],method,budget,'BASE_FP16'])
            if row is not None:
                if (row.get('source_hash')!=digest(e['inference']) or row.get('candidate_ids')!=selection['lineage']['candidate_ids'] or
                    row.get('candidate_content_hashes')!=selection['lineage']['candidate_content_hashes'] or
                    row.get('evidence_hash')!=selection['evidence_hash'] or row.get('method')!=method or row.get('budget')!=budget or
                    row.get('split')!='VALIDATION' or row.get('selected_unit_ids')!=selection['selected_unit_ids']):
                    diagnostics.append(source+': rejected source/pool/evidence/split mismatch');continue
                candidates.append((row,source))
            else:diagnostics.append(source)
    # Shared exact-prompt cache is legitimate even across methods with identical context.
    row,source=legacy_get(parent,f,'generation',[e['dataset'],e['query_id'],digest(selection['evidence_ids']),'BASE_FP16'])
    if row is not None:candidates.append((row,source))
    else:diagnostics.append(source)
    for row,source in candidates:
        if row.get('reader')!='BASE_FP16' or row.get('model_hash')!=digest([parent['models']['Qwen'],'BASE_FP16',None]) or row.get('prompt_hash')!=expected_prompt:
            diagnostics.append(source+': rejected reader/model/prompt mismatch');continue
        raw=row.get('raw_prediction',row.get('prediction'))
        if not isinstance(raw,str):diagnostics.append(source+': no raw prediction');continue
        parsed=parse_label(raw,parent['labels']) if e['task_type']=='verification' else raw
        if parsed!=row['prediction']:diagnostics.append(source+': parser mismatch');continue
        required=['prediction','prompt_tokens','generated_tokens','prompt_hash','model_hash','reader']
        if not all(k in row for k in required):diagnostics.append(source+': missing runtime identity');continue
        return {k:row[k] for k in required+['raw_prediction','generation_seconds','generation_cap_reached'] if k in row},source,diagnostics
    return None,None,diagnostics
