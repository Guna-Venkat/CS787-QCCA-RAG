"""Explicit manual phases; atomic resume; no TEST path and no policy training."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .storage import ROOT,Store,atomic,digest,lock
from .protocol import load,freeze,guard
from .encoding import Encoder,rank,unit
from .data import prepare
from .analysis import retrieval_metrics,oracle

METHODS=('BM25','MiniLM','SFR')

def roots(cfg,phase):
    base=ROOT/cfg['output_root']; base.mkdir(parents=True,exist_ok=True)
    return base,base/'runs'/phase

def identity(cfg,phase,inputs):
    print('Validating frozen source/config/data/model identity...',flush=True)
    frozen=freeze(cfg)
    extra={}
    if phase.startswith('G-'):
        from huggingface_hub import snapshot_download
        from .storage import file_hash
        spec=cfg['models'][phase[2:]]
        directory=Path(snapshot_download(spec['id'],revision=spec['revision'],local_files_only=True))
        extra={str(p):file_hash(p) for p in directory.rglob('*') if p.is_file() and p.suffix in ('.safetensors','.json','.model')}
    return dict(freeze_hash=digest(frozen),phase=phase,inputs_hash=digest(inputs),reader_weights=extra)

def run_retrieval(cfg,method):
    base,path=roots(cfg,'R0-'+method)
    with lock(path/'run.lock'):
        records=prepare(cfg); store=Store(path,identity(cfg,'R0-'+method,records))
        pending=[r for r in records if store.get(r['inference']['question_id']) is None]
        print(f'R0 {method}: {len(records)-len(pending)}/{len(records)} already saved; {len(pending)} pending',flush=True)
        if not pending:
            atomic(path/'complete.json',dict(queries=len(records),identity=store.identity));return
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(ROOT/cfg['checkpoint']),local_files_only=True)
        enc=None if method=='BM25' else Encoder(cfg['models'][method],base/'embeddings'/method,'cuda' if method=='SFR' else 'cpu')
        try:
            for i,r in enumerate(pending):
                inf=r['inference']; ranked,cost=rank(inf,method,enc)
                row=dict(question_id=inf['question_id'],paper_id=inf['paper_id'],ranked=ranked,cost=cost,
                    candidate_corpus_hash=digest(inf['chunks']),retrieval_metrics=retrieval_metrics(ranked,r['evaluation'],lambda t:len(tok.encode(t,add_special_tokens=False))))
                store.put(inf['question_id'],row)
                print(f'R0 {method}: saved {len(records)-len(pending)+i+1}/{len(records)}',flush=True)
        finally:
            if enc: enc.close()
        atomic(path/'complete.json',dict(queries=len(records),identity=store.identity))

def retrieval_rows(cfg,method,records):
    _,path=roots(cfg,'R0-'+method)
    store=Store(path,identity(cfg,'R0-'+method,records));rows={r['question_id']:r for r in store.rows()}
    if set(rows)!={r['inference']['question_id'] for r in records}:raise RuntimeError('Complete all R0 queries before generation')
    return rows

def score(result,record):
    from src.week6.evaluation import compute_qa_answer_metrics
    return compute_qa_answer_metrics(result['content'],record['evaluation']['answers'],record['evaluation']['answers_reformatted'])

def run_r(cfg,method,interface):
    records=prepare(cfg); retrieved=retrieval_rows(cfg,method,records)
    base,path=roots(cfg,f'R-{method}-{interface}')
    with lock(path/'run.lock'):
        store=Store(path,identity(cfg,f'R-{method}-{interface}',retrieved))
        pending=[(r,k) for r in records for k in cfg['k_values'] if store.get([r['inference']['question_id'],k]) is None]
        print(f'{method}/{interface}: {len(records)*len(cfg["k_values"])-len(pending)}/{len(records)*len(cfg["k_values"])} answers saved; {len(pending)} pending',flush=True)
        if not pending:
            atomic(path/'complete.json',dict(queries=len(records),answers=len(records)*len(cfg["k_values"])));return
        # SFR must be closed before loading SARA. Its R0 corpus cache already contains all chunk vectors.
        enc=Encoder(cfg['models']['SFR'],base/'embeddings/SFR','cuda') if interface=='sara' else None
        vectors={}
        try:
            if enc:
                for r,_ in pending:
                    q=r['inference']['question_id']
                    if q not in vectors:vectors[q]=enc.encode([c['text'] for c in retrieved[q]['ranked'][:10]],allow_compute=False)[0]
        finally:
            if enc:enc.close()
        from .runtime import SaraReader
        reader=SaraReader(cfg)
        try:
            for i,(r,k) in enumerate(pending):
                inf=r['inference'];q=inf['question_id']; top=retrieved[q]['ranked'][:10]
                result=reader.generate(inf['question'],top,k,vectors.get(q)); metrics=score(result,r)
                row=dict(question_id=q,paper_id=inf['paper_id'],action=f'{method}/{interface}/k{k}',k=k,
                    f1=float(metrics['token_f1']),exact_match=float(metrics['exact_match']),prediction=result['content'],actual_k=min(k,len(top)),
                    evidence_tokens=sum(len(reader.tokenizer.encode(c['text'],add_special_tokens=False)) for c in top[:k]) if hasattr(reader,'tokenizer') else None,
                    answer_type=r['evaluation'].get('question_type'),
                    selected_chunk_ids=[c['chunk_id'] for c in top[:k]],candidate_set_hash=digest(top),
                    cumulative_prompt_tokens=result['prompt_tokens'],generation=result)
                store.put([q,k],row)
                if (i+1)%10==0: print(f'{method}/{interface}: saved {len(records)*len(cfg["k_values"])-len(pending)+i+1}/{len(records)*len(cfg["k_values"])}',flush=True)
        finally:reader.close()
        atomic(path/'complete.json',dict(queries=len(records),answers=len(records)*len(cfg["k_values"])))

def compact(question,chunks,encoder,tokenizer,budget,conditioned=True):
    # Exact source sentences, selection by cosine or source lead, restored source order.
    from src.week9.senttopk_e3.sentences import spans
    sentences=[dict(text=c['text'][a:b],chunk_id=c['chunk_id'],chunk_order=j,start=a) for j,c in enumerate(chunks) for a,b in spans(c['text'])]
    if not sentences:return '',[]
    order=list(range(len(sentences)))
    if conditioned:
        v,_=encoder.encode([s['text'] for s in sentences]);q,_=encoder.encode([question],'query');scores=unit(v)@unit(q)[0]
        order=sorted(order,key=lambda i:(-float(scores[i]),i))
    chosen=[];used=0
    for i in order:
        n=len(tokenizer.encode(sentences[i]['text'],add_special_tokens=False))
        if used+n<=budget:chosen.append(i);used+=n
    chosen.sort();return '\n\n'.join(sentences[i]['text'] for i in chosen),[sentences[i] for i in chosen]

def phase_gate(cfg,phase):
    base=ROOT/cfg['output_root'];prior={'C':'R','G':'C','D':'G','P':'D'}.get(phase)
    if prior:
        p=base/f'freeze_{prior}.json'
        if not p.exists(): raise RuntimeError(f'Complete/review/freeze Phase {prior} before {phase}')
        obj=json.loads(p.read_text())
        if obj.get('GO_NEXT') is not True or obj.get('execution_freeze_hash')!=digest(freeze(cfg)):
            raise ValueError('Prior phase has no matching GO_NEXT freeze')
        return obj

def run_text_phase(cfg,phase,model_key):
    prev=phase_gate(cfg,phase);records=prepare(cfg); method=prev['retriever']
    retrieved=retrieval_rows(cfg,method,records);base,path=roots(cfg,f'{phase}-{model_key}')
    if phase=='G' and model_key not in cfg['reader_panel']:raise ValueError('Reader not in preregistered panel')
    if phase=='C' and model_key!='Mistral':raise ValueError('Compression control holds reader fixed')
    variants=['no_evidence','full','sara_k4','lead256','qcs256'] if phase=='C' else ['no_evidence','full','qcs256']
    # C latent arm uses the existing matched R-SARA output, never a new model-specific projector.
    with lock(path/'run.lock'):
        store=Store(path,identity(cfg,f'{phase}-{model_key}',retrieved))
        todo=[(r,v) for r in records for v in variants if store.get([r['inference']['question_id'],v]) is None]
        if not todo:
            atomic(path/'complete.json',dict(queries=len(records),variants=variants))
            print(f'{phase}/{model_key}: complete; no model load');return
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(ROOT/cfg['checkpoint']),local_files_only=True)
        enc=Encoder(cfg['models']['MiniLM'],base/'embeddings/MiniLM','cpu')
        evidence={}
        try:
            for r,v in todo:
                q=r['inference']['question_id'];chunks=retrieved[q]['ranked'][:10]
                text='' if v in ('no_evidence','sara_k4') else '\n\n'.join(c['text'] for c in chunks) if v=='full' else compact(r['inference']['question'],chunks,enc,tok,256,v.startswith('qcs'))[0]
                if phase=='G':
                    cp=base/'runs/C-Mistral'; cs=Store(cp,json.loads((cp/'identity.json').read_text()))
                    expected=cs.get([q,v])
                    if expected is None or expected['evidence_hash']!=digest(text):
                        raise ValueError('Reader comparison evidence differs from frozen C')
                evidence[q,v]=text
        finally:enc.close()
        from .runtime import TextReader,SaraReader
        latent={}
        if phase=='C':
            sfr=Encoder(cfg['models']['SFR'],base/'embeddings/SFR','cuda')
            try:
                for r,v in todo:
                    if v=='sara_k4':
                        q=r['inference']['question_id'];latent[q]=sfr.encode([c['text'] for c in retrieved[q]['ranked'][:10]],allow_compute=False)[0]
            finally:sfr.close()
        reader=SaraReader(cfg) if phase=='C' else TextReader(cfg['models'][model_key])
        try:
            for i,(r,v) in enumerate(todo):
                q=r['inference']['question_id']; text=evidence[q,v]
                if phase=='C':
                    if v=='sara_k4':result=reader.generate(r['inference']['question'],retrieved[q]['ranked'][:10],4,latent[q])
                    else:result=reader.generate_text(r['inference']['question'],text)
                else:result=reader.generate(r['inference']['question'],text)
                metrics=score(result,r)
                store.put([q,v],dict(question_id=q,paper_id=r['inference']['paper_id'],action=v,k=0 if v=='no_evidence' else 10,
                    f1=float(metrics['token_f1']),exact_match=float(metrics['exact_match']),generation=result,prediction=result['content'],
                    evidence_hash=digest(text),evidence_text=text,evidence_tokens_canonical=len(tok.encode(text,add_special_tokens=False)),
                    cumulative_prompt_tokens=result['prompt_tokens'],retriever=method,model=model_key,answer_type=r['evaluation'].get('question_type'),
                    source_validation_rate=1.0,answer_nll=None,grounding='UNMEASURED; exact-source evidence is not answer entailment'))
                if (i+1)%10==0:print(f'{phase}/{model_key}: {i+1}/{len(todo)} pending answers saved',flush=True)
        finally:reader.close()
        atomic(path/'complete.json',dict(queries=len(records),variants=variants))

def status(cfg):
    base=ROOT/cfg['output_root'];out={}
    for p in sorted((base/'runs').glob('*/identity.json')):
        store=Store(p.parent,json.loads(p.read_text()));out[p.parent.name]=dict(saved=len(store.rows()),complete=(p.parent/'complete.json').exists())
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--phase',choices=['audit','smoke','status','R0','R','C','G','D','RCA','P'],required=True)
    p.add_argument('--retriever',choices=METHODS,default='BM25');p.add_argument('--interface',choices=['full','sara'],default='full')
    p.add_argument('--review-phase',choices=['R','C','G']);p.add_argument('--review-reason');
    p.add_argument('--reader',choices=['Mistral','Qwen','Llama'],default='Mistral');p.add_argument('--execute',action='store_true');args=p.parse_args();cfg=load(args.config)
    if args.phase=='status':print(json.dumps(status(cfg),indent=2));return
    if args.phase=='audit':
        from .preflight import validate_artifacts
        print(json.dumps(validate_artifacts(cfg),indent=2));return
    if args.phase=='smoke':
        from .smoke import run
        print(json.dumps(run(ROOT/cfg['output_root']/'synthetic_smoke'),indent=2));return
    if not args.execute:raise RuntimeError('Manual --execute required for expensive phases')
    if args.phase=='R0':return run_retrieval(cfg,args.retriever)
    if args.phase=='R':return run_r(cfg,args.retriever,args.interface)
    if args.phase in ('C','G'):return run_text_phase(cfg,args.phase,args.reader)
    if args.phase=='P':raise RuntimeError('Policy research disabled until R/C/G/D evidence supports it; no controller implemented')
    if args.phase=='D':
        from .dataset_audit import run
        return run(cfg)
    if args.phase=='RCA':
        from .report import report,review
        if args.review_phase:
            if not args.review_reason:raise ValueError('Record evidence for the manual GO_NEXT freeze')
            return review(cfg,args.review_phase,args.retriever,args.review_reason)
        return report(cfg)

if __name__=='__main__':main()
