"""Full coverage without replacement; dataset balancing acts on loss, not sampling."""
import json,os,uuid
from collections import Counter,defaultdict
from pathlib import Path

def full_coverage(rows,accumulation=16,seed=42):
    import torch
    counts=Counter(r['dataset'] for r in rows)
    if not counts:raise ValueError('Empty TRAIN schedule')
    keys=[(r['dataset'],r['query_id']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('One corpus row per unique TRAIN query required')
    n=len(rows);weights={d:n/(len(counts)*size) for d,size in counts.items()}
    canonical=sorted(range(n),key=lambda i:keys[i])
    indices=[canonical[i] for i in torch.randperm(n,generator=torch.Generator().manual_seed(seed)).tolist()];touch=Counter();coeff=Counter()
    for start in range(0,n,accumulation):
        window=indices[start:start+accumulation]
        for d in {rows[i]['dataset'] for i in window}:touch[d]+=1
        for i in window:coeff[rows[i]['dataset']]+=weights[rows[i]['dataset']]/len(window)
    return indices,dict(policy='FULL_COVERAGE_LOSS_WEIGHTED_V1',replacement=False,seed=seed,raw_eligible_examples=dict(counts),planned_unique_seen=dict(counts),planned_repeats={d:0 for d in counts},effective_loss_weights=weights,objective_weight_mass={d:counts[d]*weights[d] for d in counts},effective_samples_per_epoch=n,planned_sample_counts=dict(counts),planned_optimizer_updates_with_dataset=dict(touch),planned_optimizer_loss_coefficient_mass=dict(coeff),optimizer_steps=(n+accumulation-1)//accumulation,note='One deterministic permutation, every eligible query once. w_d=N/(D*N_d) multiplies query-mean answer NLL. Update coefficients use actual accumulation-window size, including the short last window; these are loss coefficients, not measured gradient norms or Adam parameter changes.')

class CoverageLedger:
    """Append durable forward and update events, including failed/restarted attempts."""
    def __init__(self,path,allowed,weights):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.attempt=uuid.uuid4().hex;self.allowed=set(allowed);self.weights=weights;self.pending=[];self.sequence=0
    def append(self,event):
        data=(json.dumps(dict(attempt=self.attempt,**event),sort_keys=True)+'\n').encode();fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try:
            if os.write(fd,data)!=len(data):raise RuntimeError('Incomplete training ledger write')
            os.fsync(fd)
        finally:os.close(fd)
    def forward(self,datasets,query_ids,applied_weights):
        if not (len(datasets)==len(query_ids)==len(applied_weights)):raise ValueError('Incomplete forward metadata')
        for d,q,w in zip(datasets,query_ids,applied_weights):
            if (d,q) not in self.allowed:raise ValueError('Non-TRAIN example in gradient forward')
            if abs(w-self.weights[d])>1e-5*max(1,abs(w)):raise ValueError('Unexpected dataset loss weight')
            i=self.sequence;self.sequence+=1;self.append(dict(kind='forward',index=i,dataset=d,query_id=q,loss_weight=w));self.pending.append(i)
    def step(self,step,applied=True):
        if not self.pending:raise ValueError('Optimizer step without recorded forwards')
        self.append(dict(kind='optimizer_step',step=step,forward_indices=self.pending,applied=applied));self.pending=[]
    def summary(self):
        events=[json.loads(l) for l in self.path.read_text().splitlines()] if self.path.exists() else []
        forward={};seen=defaultdict(Counter);touch=Counter();mass=Counter();updates=0;skipped=0
        for e in events:
            if e['kind']=='forward':
                key=(e['attempt'],e['index'])
                if key in forward:raise ValueError('Duplicate forward ledger event')
                forward[key]=e;seen[e['dataset']][e['query_id']]+=1
            elif e['kind']=='optimizer_step':
                xs=[forward[e['attempt'],i] for i in e['forward_indices']]
                if e['applied']:
                    updates+=1
                    for d in {x['dataset'] for x in xs}:touch[d]+=1
                    for x in xs:mass[x['dataset']]+=x['loss_weight']/len(xs)
                else:skipped+=1
        seenkeys={(d,q) for d,s in seen.items() for q in s};missing=Counter(d for d,q in self.allowed-seenkeys)
        return dict(physical_unique_examples_seen={d:len(seen[d]) for d in self.weights},physical_forward_counts={d:sum(seen[d].values()) for d in self.weights},physical_repeats={d:sum(seen[d].values())-len(seen[d]) for d in self.weights},missing_train_examples={d:missing[d] for d in self.weights},effective_loss_weights=self.weights,applied_optimizer_steps=updates,skipped_optimizer_steps=skipped,optimizer_steps_with_dataset=dict(touch),optimizer_loss_coefficient_mass=dict(mass),attempts=len({e['attempt'] for e in events}),note='Actual durable forward consumption across all attempts, including work later replayed after a checkpoint restart. Pending forwards count as seen but not as applied updates. Loss coefficients are pre-clipping/pre-Adam; they are not gradient-norm attribution.')
