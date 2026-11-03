"""Verify and restore the immutable curated bundle from the versioned archive.

No downloads, model loads, research writes or replacement of incompatible files.
"""
import argparse,hashlib,json,os,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def restore(destination=None,archive=None,metadata=None):
 meta_path=Path(metadata or ROOT/'demo/releases/curated-demo-v1.archive.json')
 meta=json.loads(meta_path.read_text());archive=Path(archive or meta_path.parent/meta['archive']);dest=Path(destination or ROOT/'demo/replay_data/v1')
 if file_sha(archive)!=meta['sha256']:raise ValueError('Curated archive SHA256 mismatch')
 payloads={}
 with tarfile.open(archive,'r:gz') as tf:
  members=tf.getmembers();expected=set(meta['members'])|{'SHA256SUMS'}
  if {m.name for m in members}!=expected or len(members)!=len(expected):raise ValueError('Unexpected or duplicate archive members')
  for m in members:
   if not m.isfile() or Path(m.name).name!=m.name:raise ValueError('Archive must contain flat regular files only')
   data=tf.extractfile(m).read()
   if m.name!='SHA256SUMS' and hashlib.sha256(data).hexdigest()!=meta['members'][m.name]:raise ValueError('Curated member checksum mismatch')
   payloads[m.name]=data
 for name in meta['members']:
  p=dest/name
  if p.exists() and file_sha(p)!=meta['members'][name]:raise ValueError('Existing incompatible file; refusing to overwrite: '+str(p))
 receipt=json.loads(payloads['PUBLISHED.json'])
 if receipt['sha256']!=meta['members']['bundle.json'] or receipt['bundle']!='bundle.json':raise ValueError('Publication receipt does not match archive')
 # Verify all files before writing any, and preserve existing identical file mtimes.
 dest.mkdir(parents=True,exist_ok=True)
 for name in meta['members']:
  p=dest/name
  if p.exists():continue
  fd,tmp=tempfile.mkstemp(dir=dest)
  try:
   with os.fdopen(fd,'wb') as f:f.write(payloads[name]);f.flush();os.fsync(f.fileno())
   try:os.link(tmp,p)
   except FileExistsError:
    if file_sha(p)!=meta['members'][name]:raise ValueError('Concurrent incompatible curated restore')
  finally:os.unlink(tmp)
 return dict(version=meta['version'],destination=str(dest),bundle_sha256=meta['bundle_sha256'],verified_files=len(meta['members']))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path);args=parser.parse_args()
 print(json.dumps(restore(args.destination),indent=2))
