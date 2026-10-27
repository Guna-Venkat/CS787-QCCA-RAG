"""Development annotation audit and paired stochastic label diagnostic."""
import json
import numpy as np
from .storage import ROOT,Store,atomic,digest,lock
from .data import prepare
from .analysis import stability,oracle

def run(cfg):
    from .runner import phase_gate,identity,retrieval_rows,compact,score
    previous=phase_gate(cfg,'D');records=prepare(cfg);base=ROOT/cfg['output_root'];method=previous['retriever']
    retrieved=retrieval_rows(cfg,method,records)
    path=base/'runs/D-stability'
    with lock(path/'run.lock'):
        store=Store(path,identity(cfg,'D-stability',retrieved))
        # First 24 by SHA256(query identity), independent of answers/outcomes. Paper membership retained.
        sample=sorted(records,key=lambda r:digest([r['inference']['paper_id'],r['inference']['question_id'],42]))[:24]
        todo=[(r,seed,a) for r in sample for seed in (42,43,44) for a in ('no_evidence','full')
              if store.get([r['inference']['question_id'],seed,a]) is None]
        if todo:
            from .runtime import TextReader
            reader=TextReader(cfg['models']['Mistral'])
            try:
                for r,seed,a in todo:
                    inf=r['inference'];q=inf['question_id'];text='' if a=='no_evidence' else '\n\n'.join(c['text'] for c in retrieved[q]['ranked'][:10])
                    result=reader.generate(inf['question'],text,seed);metrics=score(result,r)
                    store.put([q,seed,a],dict(question_id=q,paper_id=inf['paper_id'],seed=seed,action=a,
                        f1=float(metrics['token_f1']),exact_match=float(metrics['exact_match']),generation=result))
            finally:reader.close()
        rows=store.rows();by={(r['question_id'],r['seed'],r['action']):r for r in rows}
        paired=[dict(r,delta=r['f1']-by[r['question_id'],r['seed'],'no_evidence']['f1']) for r in rows if r['action']=='full']
        audit=[]
        for r in records:
            inf=r['inference'];ev=r['evaluation'];answers=ev['answers'];words=set(' '.join(map(str,answers)).lower().split());evidence=set(' '.join(ev['gold_evidence']).lower().split())
            audit.append(dict(question_id=inf['question_id'],paper_id=inf['paper_id'],question_type=ev['question_type'],
                answer_count=len(answers),annotations=len(ev['annotations']),gold_evidence_spans=len(ev['gold_evidence']),
                answer_evidence_lexical_overlap=len(words&evidence)/max(1,len(words)),
                multiple_annotations=len(ev['annotations'])>1))
        atomic(base/'dataset_diagnostics.json',dict(annotation_audit=audit,stability=stability(paired),
            oracle_by_seed={str(seed):oracle([dict(r,k=0 if r['action']=='no_evidence' else 10) for r in rows if r['seed']==seed]) for seed in (42,43,44)},
            em_stability=stability([dict(r,delta=r['exact_match']-by[r['question_id'],r['seed'],'no_evidence']['exact_match']) for r in rows if r['action']=='full']),
            semantic_correct_but_f1_lower='UNMEASURED: requires preregistered human audit; no LLM judge invented',
            canonical_decoding='greedy; repeats are sampling diagnostics only',seeds=[42,43,44],sample=24))
        atomic(path/'complete.json',dict(queries=len(sample),calls=len(sample)*6))
