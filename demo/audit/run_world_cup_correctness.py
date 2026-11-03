"""Twelve explicitly requested LIVE_CUSTOM diagnostics, never benchmark results.

Local weights only. Resume by complete immutable condition shards. Exact-prompt
reuse is explicit; a reused row still has one conceptual answer-reader call.
"""
import sys,json,time,hashlib,argparse,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from demo.live.engine import run,parse_passages
from demo.live.resources import LiveResources
from demo.live.segmentation import documents,annotate,ADAPTED
from demo.live.protocol import CONTROLLED,assert_controlled_pair,digest
from src.week11.hierarchy import build
from src.week12.algorithm import cosine,softmax
from src.week9.retrieval import bm25
QUERY='Explain how the expansion of the 2026 World Cup changed both the group-stage structure and the total tournament workload, and explain why FIFA abandoned the originally planned three-team group format.'

def immutable(p,x):
 text=json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
 if p.exists():
  if p.read_text()!=text:raise ValueError('Incompatible existing diagnostic artifact: '+str(p))
 else:
  tmp=p.with_suffix('.tmp');tmp.write_text(text);os.link(tmp,p);tmp.unlink()

def stats(values):
 a=np.asarray(values,float)
 return dict(n=len(a),min=float(a.min()),max=float(a.max()),mean=float(a.mean()),std=float(a.std()),representative=values[:8],softmax_tau5_largest_weight=max(softmax(values)))

def main():
 args=argparse.ArgumentParser();args.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=args.parse_args()
 out=Path(__file__).parent/'world_cup_correctness_v2';out.mkdir(exist_ok=True)
 from demo.live.preset import load_preset
 passages=load_preset()['passages'];r=LiveResources(args.device);tok=r.tokenizer()
 meta=dict(result_type='LIVE_CUSTOM',query=QUERY,passages=passages,segmentation=ADAPTED,scoring=CONTROLLED,device=args.device,models=r.cfg['models'],generation=r.cfg['generation'],seed=r.cfg['seed'],expected_conditions=12,code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['demo/live/engine.py','demo/live/resources.py','demo/live/protocol.py','demo/live/segmentation.py','src/week12/reader.py','src/week12/algorithm.py']})
 immutable(out/'manifest.json',meta);identity=digest(meta);started=time.monotonic();prepared={};calls={}
 # Warm all required source embeddings together, then unload SFR before reader.
 pending=[(rt,m,b) for rt in ('SFR','BM25') for m in ('RAW_SKIP','FIXED_SENTENCE','QBGF_FULL') for b in (256,1024) if not (out/f'{rt}_{m}_{b}.json').exists()]
 if pending or not (out/'score_distributions.json').exists():
  units=annotate(build(documents(passages),tok,{'max_passage_tokens':256}),ADAPTED)
  parents=[u for u in units if u['unit_type']=='PASSAGE'];fine=[u for u in units if u['unit_type']=='SENTENCE']
  print('Encoding World Cup query, 15 passages and 63 source units with pinned SFR...',flush=True)
  q=r.embed([QUERY],'query')[0];pvec=r.embed([u['source_text'] for u in parents],'document');fvec=r.embed([u['source_text'] for u in fine],'document')
  dense=cosine(q,pvec);lex=bm25(QUERY,[u['source_text'] for u in parents]).tolist()
  raw={}
  for rt,ps in [('SFR',dense),('BM25',lex)]:
   ix=sorted(range(len(parents)),key=lambda i:(-ps[i],i))[:10];ids={parents[i]['unit_id'] for i in ix};fi=[i for i,u in enumerate(fine) if u['passage_id'] in ids]
   vals=cosine(q,fvec[fi]) if rt=='SFR' else bm25(QUERY,[fine[i]['source_text'] for i in fi]).tolist()
   raw[rt+'_passages']=dict(stats=stats([ps[i] for i in ix]),values=[dict(passage=parents[i]['document_id'],score=ps[i]) for i in ix])
   raw[rt+'_units']=dict(stats=stats(vals),values=[dict(unit_id=fine[i]['unit_id'],passage=fine[i]['document_id'],text=fine[i]['source_text'],score=float(v)) for i,v in zip(fi,vals)])
  immutable(out/'score_distributions.json',dict(identity=identity,distributions=raw,note='Native score distributions reproduce old allocator inputs for audit only. Corrected live BM25-native QBGF is blocked. No native BM25-QBGF answer is generated.'))
  r.release_gpu();print('SFR unloaded; embedding arrays retained. Starting/resuming requested 12 answer conditions.',flush=True)
 original_generate=r.generate
 def cached_generate(query,selection):
  k=digest([identity,query,selection['evidence_ids']]);path=out/('answer_cache_'+k+'.json')
  if path.exists():
   x=json.loads(path.read_text());x['exact_prompt_cache_reused']=True;return x
  x=original_generate(query,selection);immutable(path,x);x['exact_prompt_cache_reused']=False;return x
 r.generate=cached_generate
 results=[]
 try:
  for rt in ('SFR','BM25'):
   for m in ('RAW_SKIP','FIXED_SENTENCE','QBGF_FULL'):
    for b in (256,1024):
     path=out/f'{rt}_{m}_{b}.json'
     if path.exists():
      value=json.loads(path.read_text());assert value['diagnostic_identity']==identity
     else:
      print(f'Start {rt}/{m}/{b}; elapsed={time.monotonic()-started:.0f}s',flush=True)
      value=run(QUERY,passages,m,b,rt,r,True,prepared_cache=prepared,segmentation=ADAPTED,scoring_mode=CONTROLLED)
      if value['answer'] is None:raise RuntimeError(value['reader_error'])
      value['diagnostic_identity']=identity;value['conceptual_lm_calls']=1
      immutable(path,value)
     results.append(value);print(f'Saved {len(results)}/12 {rt}/{m}/{b}; evidence={value["selection"]["evidence_tokens"]}; elapsed={time.monotonic()-started:.0f}s',flush=True)
 finally:r.release_gpu()
 for m in ('RAW_SKIP','FIXED_SENTENCE','QBGF_FULL'):
  for b in (256,1024):assert_controlled_pair(*[next(x for x in results if x['retriever']==rt and x['method']==m and x['budget']==b) for rt in ('BM25','SFR')])
 patterns={'48-team expansion':r'48 teams','12 groups of four':r'12 groups of 4','original 16 groups of three':r'16 groups of 3','64 to 104 matches':r'64 to 104','7 to 8 semifinalist matches':r'7 to 8','39-day duration':r'39 days','collusion concern':r'collusion','penalty shootouts':r'penalty shootouts'}
 lines=['# World Cup LiveDemo corrective regression — 12 conditions','', 'These are custom-input diagnostics, not benchmark results. User-supplied source assertions are not fact-checked. No gold scores or answer-quality winner are inferred.','',f'Query: {QUERY}', '',f'Device: {args.device}; common pinned SFR downstream scoring; segmentation: {ADAPTED}. Both candidate retrievers use identical method, budget, reader, prompt template, parser and decoding. Exact prompt contents differ when evidence differs.','', '| Candidate retriever | Method | Budget | Evidence tokens | Prompt tokens | Generated tokens |','|---|---|---:|---:|---:|---:|']
 for v in results:lines.append(f"| {v['retriever']} | {v['method']} | {v['budget']} | {v['selection']['evidence_tokens']} | {v['prompt_tokens']} | {v['generated_tokens']} |")
 for v in results:
  lines += ['',f"## {v['retriever']} · {v['method']} · {v['budget']}",'', 'Top-10 IDs: '+', '.join(p['document_id'] for p in v['pool']), '', '**Answer**', '',v['answer'],'','**Exact selected evidence**','','```text',v['selection']['text'],'```','', 'Unit types: '+', '.join(u.get('display_kind',u['unit_type']) for u in v['selection']['units']),'', '**Descriptive phrase inspection (not a metric)**','', '| Source concept | Candidate-pool passages containing phrase | Selected units containing phrase |','|---|---|---|']
  for concept,pat in patterns.items():
   pp=[p['document_id'] for p in v['pool'] if __import__('re').search(pat,p['source_text'],__import__('re').I)]
   uu=[u['unit_id'] for u in v['selection']['units'] if __import__('re').search(pat,u['source_text'],__import__('re').I)]
   lines.append('| '+concept+' | '+(', '.join(pp) or 'No literal match')+' | '+(', '.join(uu) or 'No literal match')+' |')
  if v['qbgf_options']:
   lines+=['','**Chosen QBGF options**','','| Passage | Option | Cost | Utility |','|---|---|---:|---:|']
   for o in v['qbgf_options']:
    if o['selected']:lines.append(f"| {o['passage_id']} | {o['label']} | {o['cost']} | {o['utility']:.6f} |")
 lines+=['','Full ranking, selected IDs/text, evidence types, exact token IDs, all QBGF options, per-unit selected status and invariant metadata are retained in each condition JSON. Previous pre-fix answers are not a method-quality baseline; any difference can result from segmentation/scoring changes.']
 (out/'WORLD_CUP_12_CONDITION_REPORT.md').write_text('\n'.join(lines)+'\n')
 if not (out/'COMPLETE.json').exists():immutable(out/'COMPLETE.json',dict(identity=identity,conditions=len(results),pairs_verified=6,seconds=time.monotonic()-started))
 else:
  receipt=json.loads((out/'COMPLETE.json').read_text());assert receipt['identity']==identity and receipt['conditions']==len(results) and receipt['pairs_verified']==6
 print('COMPLETE: 12 conditions, six controlled pairs verified. No benchmark scores computed.',flush=True)
if __name__=='__main__':main()
