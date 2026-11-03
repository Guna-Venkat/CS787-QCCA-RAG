"""Read-only integrity snapshot of tracked research plus published demo identities."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SNAP=Path(__file__).with_name('correctness_protected_before.json')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def snapshot():
 names=subprocess.check_output(['git','ls-files','results/week11','results/week12','src/week11','src/week12','src/week9/senttopk_e3'],cwd=ROOT,text=True).splitlines()
 names+=['demo/replay_data/v1/'+x for x in ('bundle.json','PUBLISHED.json','manifest.json')]
 return {n:sha(ROOT/n) for n in sorted(set(names)) if (ROOT/n).is_file()}
if __name__=='__main__':
 import sys
 if '--verify' in sys.argv:
  expected=json.loads(SNAP.read_text());actual=snapshot();changed=[n for n,h in expected.items() if actual.get(n)!=h]
  if changed:raise RuntimeError('Protected artifacts changed: '+str(changed))
  print('Verified unchanged:',len(expected),'protected files')
 else:
  if SNAP.exists():raise RuntimeError('Baseline exists; use --verify')
  data=snapshot();SNAP.write_text(json.dumps(data,indent=2)+'\n');print('Recorded',len(data),'protected files')
