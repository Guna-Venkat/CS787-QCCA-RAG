"""Offline read-only C/D aggregation; no generation or evidence writes."""
from reportlib import *
import collections,numpy as np

def main():
 out=BASE/'week_12';out.mkdir(exist_ok=True);groups={};seen={};dup=0;conflicts=0;counts={};labels=['SUPPORT','REFUTE','NOT_ENOUGH_INFO'];source_receipts=[]
 for phase in ['C','D']:
  folder=ROOT/f'results/week12/qbgf/v3/validation/{phase}/rows';n=0;manifest=hashlib.sha256()
  for p in sorted(folder.glob('*.json')):
   if p.name=='identity.json':continue
   v=json.loads(p.read_text());r=v['row'];n+=1;manifest.update((p.name+':'+v['checksum']+'\n').encode())
   key=(r['dataset'],str(r['query_id']),r['method'],str(r['budget']),r['reader']);finger=(r['prediction'],r['evidence_hash'],r['source_hash'])
   if key in seen:
    dup+=1;conflicts+=int(seen[key]!=finger);continue
   seen[key]=finger;gkey=(r['dataset'],r['method'],str(r['budget']),r['reader']);g=groups.setdefault(gkey,dict(n=0,f1=0.,prompt=0.,tp=np.zeros(3),fp=np.zeros(3),fn=np.zeros(3)))
   g['n']+=1;g['prompt']+=r['prompt_tokens']
   if r['dataset']=='SciFact':
    for i,label in enumerate(labels):
     g['tp'][i]+=r['gold']==label and r['prediction']==label;g['fp'][i]+=r['gold']!=label and r['prediction']==label;g['fn'][i]+=r['gold']==label and r['prediction']!=label
   else:g['f1']+=r['f1']
  counts[phase]=n;source_receipts.append(dict(phase=phase,rows=n,manifest_sha256_of_sorted_names_and_stored_checksums=manifest.hexdigest(),source_directory=str(folder.relative_to(ROOT))))
 stored=frame('results/week12/qbgf/v3/analysis/summary.csv');rows=[]
 for key,g in groups.items():
  d,m,b,r=key
  if d=='SciFact':
   den=2*g['tp']+g['fp']+g['fn'];quality=np.divide(2*g['tp'],den,out=np.zeros(3),where=den!=0).mean()
  else:quality=g['f1']/g['n']
  x=stored[(stored.dataset==d)&(stored.method==m)&(stored.budget.astype(str)==b)&(stored.reader==r)];assert len(x)==1
  delta=float(quality-x.quality.iloc[0]);assert abs(delta)<1e-10 and g['n']==x.n_queries.iloc[0]
  rows.append(dict(dataset=d,method=m,budget=b,reader=r,n=g['n'],recomputed_quality=float(quality),stored_quality=float(x.quality.iloc[0]),difference=delta,recomputed_prompt_tokens=g['prompt']/g['n']))
 pd.DataFrame(rows).to_csv(out/'OfflineValidationRecomputation.csv',index=False)
 audit=dict(raw_counts=counts,unique_rows=len(seen),duplicate_C_D_rows=dup,conflicting_duplicates=conflicts,conditions=len(rows),all_means_match=True,source_receipts=source_receipts,source_summary_sha=sha(ROOT/'results/week12/qbgf/v3/analysis/summary.csv'))
 (out/'OfflineValidationRecomputation.json').write_text(json.dumps(audit,indent=2)+'\n');assert not conflicts;print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
