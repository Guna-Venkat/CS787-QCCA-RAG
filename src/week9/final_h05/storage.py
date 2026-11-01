"""Atomic per-query shards; incompatible resumes and torn rows never merge."""
import json
import os
import tempfile
from pathlib import Path
from ..final_h import protocol as p


def atomic_bytes(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def atomic_json(path, value):
    atomic_bytes(path,(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode())


class Shards:
    def __init__(self,root,identity):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.identity=identity
        path=self.root/'identity.json'
        if path.exists():p.require(p.read(path)==identity,'Resume identity/config/source mismatch')
        else:atomic_json(path,identity)
    def path(self,q):return self.root/(p.digest(str(q))+'.json')
    def get(self,q):
        path=self.path(q)
        if not path.exists():return None
        row=p.read(path)
        p.require(row['identity_hash']==p.digest(self.identity) and row['question_id']==str(q),'Shard identity differs')
        p.require(row['checksum']==p.digest(row['payload']),'Shard checksum differs')
        return row['payload']
    def put(self,q,value):
        old=self.get(q)
        if old is not None:
            p.require(old==value,'Completed shard cannot be overwritten');return old
        atomic_json(self.path(q),dict(question_id=str(q),identity_hash=p.digest(self.identity),payload=value,checksum=p.digest(value)))
        return value


def membership(rows):
    keys=[(str(r['question_id']),str(r['paper_id'])) for r in rows]
    p.require(bool(keys) and len(keys)==len(set(keys)) and len({q for q,_ in keys})==len(keys),'Duplicate/empty query membership')
    p.require(all(r['split']=='test' for r in rows),'Split contamination')
    return sorted(keys)


def pair_membership(left,right):
    p.require(membership(left)==membership(right),'Query/paper membership differs; cannot pair')
