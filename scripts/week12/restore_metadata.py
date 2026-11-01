"""Restore exact lossless metadata copies; never overwrite conflicting bytes."""
from pathlib import Path
import gzip,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.week12.protocol import guard

def main():
 guard()
 for p in (ROOT/'results/week12').rglob('*.json.gz'):
  target=p.with_suffix('');data=gzip.decompress(p.read_bytes())
  if target.exists():
   if target.read_bytes()!=data:raise ValueError('Refusing to overwrite different immutable metadata: '+str(target))
  else:target.write_bytes(data)
  print('Verified/restored',target.relative_to(ROOT))
if __name__=='__main__':main()
