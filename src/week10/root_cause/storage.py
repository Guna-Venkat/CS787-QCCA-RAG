"""Atomic immutable query checkpoints; completed work is never regenerated."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[3]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(name): os.unlink(name)

@contextlib.contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This phase already has an active writer')
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)

class Store:
    def __init__(self, path, identity):
        self.path = Path(path); self.path.mkdir(parents=True, exist_ok=True)
        p = self.path/'identity.json'
        if p.exists() and json.loads(p.read_text()) != identity:
            raise ValueError('Provenance changed: use a NEW version, never mix or replace saved work')
        if not p.exists(): atomic(p, identity)
        self.identity = digest(identity)
    def filename(self, key): return self.path/(digest(key)+'.json')
    def get(self, key):
        p = self.filename(key)
        if not p.exists(): return None
        obj = json.loads(p.read_text())
        if obj['identity'] != self.identity or obj['key'] != key or digest(obj['row']) != obj['checksum']:
            raise ValueError('Corrupt or mismatched query checkpoint: '+str(p))
        return obj['row']
    def put(self, key, row):
        old = self.get(key)
        if old is not None:
            if old != row: raise ValueError('Attempt to overwrite a completed query')
            return old
        atomic(self.filename(key), dict(key=key, identity=self.identity, checksum=digest(row), row=row))
        return row
    def rows(self):
        out=[]
        for p in sorted(self.path.glob('*.json')):
            if len(p.stem)!=64: continue
            obj=json.loads(p.read_text()); out.append(self.get(obj['key']))
        return out
