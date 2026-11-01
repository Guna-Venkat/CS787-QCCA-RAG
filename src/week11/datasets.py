"""Native local adapters. All labels are isolated from inference dictionaries."""
import json
from pathlib import Path
from .protocol import guard, ROOT
from .cache import digest, file_hash


def document(dataset, doc_id, text, sentences=None, passages=None):
    return dict(dataset=dataset, document_id=str(doc_id), text=text,
                native_sentences=sentences or [], native_passages=passages or [])


def join_sentences(values):
    text=''; spans=[]
    for index,value in enumerate(values):
        if text:text+='\n'
        start=len(text);text+=value
        spans.append(dict(start_char=start,end_char=len(text),native_id=index))
    return text,spans


def normalize(dataset, split, qid, question, documents, answer=None, label=None, gold=None, answer_type=None, metadata=None):
    guard(dataset,split)
    task={'QASPER':'freeform_qa','HotpotQA':'multihop_qa','SciFact':'verification'}[dataset]
    did=documents[0]['document_id'] if len(documents)==1 else str(qid)
    ev=dict(gold_answer=answer, gold_label=label, gold_evidence_ids=[g['evidence_id'] for g in gold or []],
            gold_evidence_text=[g['text'] for g in gold or []], gold_spans=gold or [], answer_type=answer_type)
    inf=dict(dataset=dataset, split=split, query_id=str(qid), document_id=did,
             question=question, documents=documents, task_type=task)
    # Metadata used by inference must never carry evaluation fields.
    return dict(dataset=dataset,split=split,query_id=str(qid),document_id=did,paper_id=did,
        question=question,gold_answer=answer,gold_label=label,gold_evidence_ids=ev['gold_evidence_ids'],
        gold_evidence_text=ev['gold_evidence_text'],document_text='\n\n'.join(d['text'] for d in documents),
        metadata=metadata or {},answer_type=answer_type,task_type=task,inference=inf,evaluation=ev,
        cluster_id=did if dataset=='QASPER' else str(qid))


def inference(example):
    guard(example['dataset'], example['split'])
    return example['inference']


def hotpot(rows, split='dev'):
    result=[]
    for row in rows:
        docs=[]; spans={}
        for title,sentences in row['context']:
            body,ss=join_sentences(sentences);prefix=title+'\n';text=prefix+body
            ss=[dict(s,start_char=s['start_char']+len(prefix),end_char=s['end_char']+len(prefix)) for s in ss]
            did=digest(['HotpotQA',title,text])[:24]
            docs.append(document('HotpotQA',did,text,[dict(start_char=0,end_char=len(title),native_id=-1)]+ss,[dict(start_char=0,end_char=len(text))]))
            for s in ss:spans[title,s['native_id']]=dict(document_id=did,start_char=s['start_char'],end_char=s['end_char'],
                evidence_id=f'{title}:{s["native_id"]}',text=text[s['start_char']:s['end_char']])
        gold=[]
        for title,index in row.get('supporting_facts',[]):
            gold.append(spans.get((title,index),dict(document_id=None,start_char=None,end_char=None,
                evidence_id=f'{title}:{index}',text='',unmapped=True)))
        result.append(normalize('HotpotQA',split,row['_id'],row['question'],docs,[row['answer']],gold=gold,
            answer_type=row.get('type','unknown'),metadata=dict(corpus='provided distractor context',cluster='query')))
    return result


def scifact(claims, corpus, split='dev'):
    docs={}
    for row in corpus:
        body,ss=join_sentences(row['abstract']);prefix=row['title']+'\n';text=prefix+body;did=str(row['doc_id'])
        ss=[dict(s,start_char=s['start_char']+len(prefix),end_char=s['end_char']+len(prefix)) for s in ss]
        docs[did]=document('SciFact',did,text,[dict(start_char=0,end_char=len(row['title']),native_id=-1)]+ss,[dict(start_char=0,end_char=len(text))])
    result=[]
    for row in claims:
        gold=[];labels=set()
        for did,groups in row.get('evidence',{}).items():
            for group in groups:
                labels.add({'SUPPORT':'SUPPORT','CONTRADICT':'REFUTE'}[group['label']])
                for index in group['sentences']:
                    d=docs[str(did)];s=next(s for s in d['native_sentences'] if s['native_id']==index)
                    gold.append(dict(document_id=str(did),start_char=s['start_char'],end_char=s['end_char'],
                        evidence_id=f'{did}:{index}',text=d['text'][s['start_char']:s['end_char']]))
        if len(labels)>1:raise ValueError('Conflicting SciFact labels require explicit policy')
        label=next(iter(labels)) if labels else 'NOT_ENOUGH_INFO'
        # Full corpus, never oracle cited/gold documents as the candidate corpus.
        result.append(normalize('SciFact',split,row['id'],row['claim'],list(docs.values()),label=label,gold=gold,
            answer_type=label,metadata=dict(corpus='all SciFact corpus abstracts',cluster='claim',native_evidence=row.get('evidence',{}))))
    return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_dataset(name, cfg):
    spec=cfg['datasets'][name];guard(name,spec['split'])
    if name=='QASPER':
        from .qasper_adapter import load as qasper_load
        return qasper_load(spec)
    path=ROOT/spec['path']
    if not path.exists():raise FileNotFoundError(f'{path}: run the explicit Week-11 data preparation command first')
    if name=='HotpotQA':return hotpot(json.loads(path.read_text()),spec['split'])
    return scifact(read_jsonl(path),read_jsonl(ROOT/spec['corpus_path']),spec['split'])


def source_identity(cfg):
    sources={}
    for name,spec in cfg['datasets'].items():
        guard(name,spec['split'])
        for key in ('path','raw_path','corpus_path'):
            if key in spec:
                p=ROOT/spec[key]
                if name=='QASPER' and 'test' in p.name.lower():raise ValueError('QASPER TEST path blocked')
                if not p.exists():raise FileNotFoundError(p)
                sources[str(p)]=file_hash(p)
    return sources
