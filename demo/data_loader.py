"""Primary artifact readers. Never instantiate the experiment Store (which writes)."""
import csv, hashlib, json, math
from functools import lru_cache
from pathlib import Path
from demo.constants import ROOT, SOURCES, DATASETS, BUDGETS, FREEZE_SHA

class DataError(ValueError): pass

def path_for(name):
    rel = SOURCES.get(name, name)
    p = (ROOT / rel).resolve()
    if not p.is_relative_to(ROOT): raise DataError('Source must be inside this repository')
    if p.suffix.lower() not in ('.csv', '.json', '.py'): raise DataError('Only primary CSV/JSON artifacts may supply data')
    return p

def _storage(path):
    p=Path(path)
    if not p.exists() and p.suffix=='.json' and p.with_suffix('.json.gz').exists():return p.with_suffix('.json.gz')
    return p

def _source_bytes(path):
    import gzip
    p=Path(path);stored=_storage(p)
    data=stored.read_bytes()
    return gzip.decompress(data) if stored!=p else data

def sha(path): return hashlib.sha256(_source_bytes(path)).hexdigest()

def signature(name):
    p=path_for(name)
    stored=_storage(p)
    return (str(p),stored.stat().st_mtime_ns,stored.stat().st_size) if stored.exists() else (str(p),None,None)

@lru_cache(maxsize=48)
def _load(path, mtime, size):
    p=Path(path)
    if not _storage(p).exists(): raise DataError('Missing primary source: '+str(p.relative_to(ROOT)))
    if p.suffix=='.json': return json.loads(_source_bytes(p))
    with p.open(newline='') as f:
        rows=[]
        for row in csv.DictReader(f):
            for k,v in row.items():
                if v is None or v=='': row[k]=None
                elif v in ('True','False'): row[k]=v=='True'
                else:
                    try:
                        n=float(v)
                        row[k]=n if math.isfinite(n) else None
                    except (ValueError,TypeError): pass
            # IDs can look numeric; preserve their exact serialized form.
            rows.append(row)
    with p.open(newline='') as f:
        raw=list(csv.DictReader(f))
    for row,original in zip(rows,raw):
        for k in ('query_id','unit_id','cluster_id','id','paper_id'):
            if k in original:row[k]=original[k]
    return rows

def load(name, optional=False):
    if path_for(name).suffix not in ('.csv','.json'):raise DataError('Definitions are provenance only, not numerical data')
    sig=signature(name)
    if sig[1] is None and optional:return None
    return _load(*sig)

def validate_rows(rows, kind='summary'):
    seen=set()
    for r in rows:
        if r.get('dataset') not in DATASETS:raise DataError('Unknown dataset')
        if r.get('budget') not in BUDGETS:raise DataError('Unknown evidence budget')
        if not isinstance(r.get('method'),str):raise DataError('Missing method identity')
        if kind=='summary':
            if r.get('quality') is None or not 0<=r['quality']<=1:raise DataError('Missing/invalid quality metric')
            key=(r['dataset'],r['method'],r['budget'])
            if key in seen:raise DataError('Duplicate condition')
            seen.add(key)
            metric=r.get('primary_metric',r.get('metric'))
            valid=('macro_f1','macro_F1') if r['dataset']=='SciFact' else ('f1','answer_F1')
            if metric not in valid:raise DataError('Metric does not match dataset')
    return rows

def summaries(name): return validate_rows(load(name))

def freeze_check():
    paths=[path_for('freeze'),path_for('freeze_copy')]
    if any(not p.exists() or sha(p)!=FREEZE_SHA for p in paths):raise DataError('Part-I freeze identity mismatch')
    return FREEZE_SHA

def provenance(name):
    p=path_for(name)
    return {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':len(_source_bytes(p))}

def manifest():
    return {k:provenance(k) for k in SOURCES if _storage(path_for(k)).exists() and path_for(k).suffix in ('.csv','.json')}

@lru_cache(maxsize=64)
def _fingerprint(path,mtime,size):return sha(path)

def verify_manifest():
    p=ROOT/'demo/artifact_manifest.json'
    if not p.exists():raise DataError('Demo provenance manifest missing')
    expected=json.loads(p.read_text())['sources']
    for name,entry in expected.items():
        actual=path_for(name)
        if not _storage(actual).exists():raise DataError('Registered source unavailable: '+entry['path'])
        sig=signature(name)
        if _fingerprint(*sig)!=entry['sha256']:raise DataError('Registered source changed: '+entry['path'])
    return True
