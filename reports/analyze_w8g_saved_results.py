"""Recompute W8-G aggregates and paired paper bootstrap from saved predictions."""
from pathlib import Path
import csv, hashlib, json, random
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/week8/w8g_lora/full'
variants = ['original_SARA', 'uniform_SFT', 'HC_LoRA', 'OW_LoRA']
records = {}
summary = json.loads((BASE/'summary.json').read_text())
for variant in variants:
    path = BASE/f'{variant}_predictions.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == summary['artifact_sha256'][path.name]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    records[variant] = {(r['context'],r['question_id']):r for r in rows}
    assert len(rows)==len(records[variant])==684
    for context in ['k4','frozen_controller']:
        group=[r for r in rows if r['context']==context]
        assert len(group)==342
        mean=sum(r['token_f1'] for r in group)/len(group)
        assert abs(mean-summary['results'][variant][context]['downstream_f1'])<1e-12
base=records['original_SARA']
for variant in variants[1:]:
    assert records[variant].keys()==base.keys()
    for key,row in records[variant].items():
        reference=base[key]
        for field in ['paper_id','selected_action','candidate_passage_index','full_indices','compressed_indices','final_prompt_tokens','cumulative_prompt_tokens']:
            assert row[field]==reference[field],(variant,key,field)
        if key[0]=='frozen_controller' and row['selected_action']=='STOP':
            assert row['prediction']==reference['prediction']
pairs=[]
for context in ['k4','frozen_controller']:
    for variant in variants[1:]:
        pairs.append((variant,context,'original_SARA',context))
    for variant in ['HC_LoRA','OW_LoRA']:
        pairs.append((variant,context,'uniform_SFT',context))
for variant in variants[1:]:
    pairs.append((variant,'frozen_controller',variant,'k4'))
output=[]
for va,ca,vb,cb in pairs:
    grouped=defaultdict(list)
    for (context,qid),row in records[va].items():
        if context==ca:
            other=records[vb][(cb,qid)]
            assert row['paper_id']==other['paper_id']
            grouped[row['paper_id']].append(row['token_f1']-other['token_f1'])
    totals=[(sum(v),len(v)) for _,v in sorted(grouped.items())]
    rng=random.Random(42); boot=[]
    for _ in range(10000):
        sample=rng.choices(totals,k=len(totals))
        boot.append(sum(s for s,n in sample)/sum(n for s,n in sample))
    boot.sort()
    def quantile(p):
        pos=(len(boot)-1)*p; i=int(pos); return boot[i]+(boot[min(i+1,len(boot)-1)]-boot[i])*(pos-i)
    output.append(dict(comparison=f'{va} {ca} vs {vb} {cb}',queries=342,papers=len(totals),delta_f1=sum(s for s,n in totals)/342,ci95_lower=quantile(.025),ci95_upper=quantile(.975),bootstrap_resamples=10000,seed=42))
path=ROOT/'reports/w8g_paired_paper_bootstrap.csv'
with path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
print(json.dumps(output,indent=2))
print('Verified prediction hashes, 684 unique rows per variant, means, fixed actions/contexts/tokens and original STOP answers.')
