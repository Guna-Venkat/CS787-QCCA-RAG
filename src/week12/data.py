"""Conservative split audit. No TEST payloads are opened. Unverified history blocks confirmation."""
import json,re
from pathlib import Path
from collections import defaultdict
from .protocol import ROOT,root,digest,sha,immutable,atomic,guard,utc

def qhash(q):return digest(' '.join(q.lower().split()))
def components(records):
    # Union claims sharing cited papers, including transitive links; grouping only, not selection.
    parent={r['query_id']:r['query_id'] for r in records}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    owners={}
    for r in records:
        for paper in r.get('split_documents',[]):
            if paper in owners:parent[find(r['query_id'])]=find(owners[paper])
            owners[paper]=r['query_id']
    return {q:find(q) for q in parent}

def w11_examples():
    from src.week11.protocol import load
    from src.week11.datasets import load_dataset
    c=load();return c,{d:load_dataset(d,c) for d in c['datasets']}
def native_qasper(path,allowed_ids=None):
    guard('QASPER','validation',path)
    from datasets import Dataset
    from src.week11.datasets import normalize,document
    from src.week10.root_cause.data import evidence_occurrences
    out=[]
    for paper in Dataset.from_file(str(path)):
        pid=str(paper['id']);context=[paper['title']+'\t'+paper['abstract']]
        for name,paras in zip(paper['full_text']['section_name'],paper['full_text']['paragraphs']):
            if paras and paras!=['']:context.append(re.sub(r'BIBREF\d+','',str(name)+'\t'+' '.join(paras)))
        text='\n\n'.join(context)
        for j,q in enumerate(paper['qas']['question']):
            qid=str(paper['qas']['question_id'][j])
            if allowed_ids is not None and qid not in allowed_ids:continue
            refs=[];evidence=set();types=[]
            for a in paper['qas']['answers'][j]['answer']:
                v='Unanswerable' if a['unanswerable'] else 'Yes' if a['yes_no'] is True else 'No' if a['yes_no'] is False else a['free_form_answer'] or ', '.join(a['extractive_spans'])
                if v:refs.append(v)
                evidence.update(t for t in a.get('evidence',[]) if t.strip());types.append('unanswerable' if a['unanswerable'] else 'yes_no' if a['yes_no'] is not None else 'free_form' if a['free_form_answer'] else 'extractive')
            spans=[]
            for i,(t,occ) in enumerate(zip(sorted(evidence),evidence_occurrences(text,sorted(evidence)))):
                spans.append(dict(document_id=pid,evidence_id=f'{pid}:gold{i}',text=t,alternatives=occ,start_char=occ[0][0] if occ else None,end_char=occ[0][1] if occ else None,unmapped=not bool(occ)))
            ex=normalize('QASPER','dev',qid,q,[document('QASPER',pid,text)],refs or ['Unanswerable'],gold=spans,answer_type=types[0] if len(set(types))==1 else 'mixed',metadata={'official_split':path.stem,'native_paper_id':pid,'canonical_context':context});out.append(ex)
    return out

def audit(cfg):
    if cfg.get('split_protocol')=='TRAIN_VAL_CONFIRM_v2':
        from .splits import audit as corrected_audit
        return corrected_audit(cfg)
    c,all_examples=w11_examples();out=root(cfg)/'protocol';out.mkdir(parents=True,exist_ok=True)
    # Project QASPER TRAIN is raw official TRAIN; all these questions are historically exposed.
    used={d:set() for d in cfg['datasets']};used_questions={d:set() for d in cfg['datasets']};used_docs={d:set() for d in cfg['datasets']}
    sources=[]
    qpaths=[ROOT/'Baselines/SARA-main/data/reformatted'/f for f in ['QASPER_train.jsonl','QASPER_train_split.jsonl','QASPER_dev.jsonl']]
    for p in qpaths:
        guard('QASPER','fit',p);sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),role='historical exposure: conservatively all rows'))
        for line in p.read_text().splitlines():
            r=json.loads(line);used['QASPER'].add(str(r['id']));used_questions['QASPER'].add(qhash(r['question']));used_docs['QASPER'].add(str(r['example_id']))
    cohort_path=ROOT/'results/week11/multigranular/v1/granularity_budget/cohort.json';old=json.loads(cohort_path.read_text());pilot=old.get('selected',old.get('queries',old))
    # Cohort actual schema inspected without model/answer reads.
    if isinstance(pilot,dict):
        ids={d:set(map(str,pilot.get(d,[]))) for d in cfg['datasets']}
        if not any(ids.values()):
            values=old.get('query_ids',old.get('members',[]));ids={d:set(str(v['query_id']) for v in values if v['dataset']==d) for d in cfg['datasets']}
    else:ids={d:set(str(v['query_id']) for v in pilot if v['dataset']==d) for d in cfg['datasets']}
    # Fallback only to the exact saved C baseline rows; no outcome is consulted.
    if not all(ids.values()):
        ids={d:set() for d in cfg['datasets']}
        for p in (cohort_path.parent/'rows').glob('*.json'):
            if len(p.stem)!=64:continue
            key=json.loads(p.read_text())['key'];ids[key[0]].add(str(key[1]))
    development={};confirmation={};report=[];training={};candidate_records=[]
    for dataset,examples in all_examples.items():
        dev=[e for e in examples if e['query_id'] in ids[dataset]];assert len(dev)==50,(dataset,len(dev))
        for e in dev:
            used[dataset].add(e['query_id']);used_questions[dataset].add(qhash(e['question']));used_docs[dataset].update(d['document_id'] for d in e['inference']['documents'])
        development[dataset]=[dict(query_id=e['query_id'],paper_id=e['paper_id'],question_hash=qhash(e['question'])) for e in dev]
        candidates=[e for e in examples if e['query_id'] not in used[dataset] and qhash(e['question']) not in used_questions[dataset]]
        if dataset=='QASPER':
            vp=Path(c['datasets']['QASPER']['raw_path']).with_name('qasper-validation.arrow')
            candidates=native_qasper(vp) if vp.exists() else []
            candidates=[e for e in candidates if qhash(e['question']) not in used_questions[dataset]]
        elif dataset=='SciFact':
            from src.week11.datasets import read_jsonl
            native=read_jsonl(ROOT/c['datasets'][dataset]['path']);groups=components([dict(query_id=str(r['id']),split_documents=[str(i) for i in r.get('cited_doc_ids',[])]) for r in native]);blocked={groups[q] for q in ids[dataset]}
            candidates=[e for e in candidates if groups[e['query_id']] not in blocked]
        candidates.sort(key=lambda e:digest([dataset,e['paper_id'] if dataset=='QASPER' else e['query_id'],cfg['seed']]))
        candidates=candidates[:cfg['confirmation_counts'][dataset]]
        confirmation[dataset]=[dict(query_id=e['query_id'],paper_id=e['paper_id'],question_hash=qhash(e['question']),status='CANDIDATE_HISTORY_REVIEW_REQUIRED') for e in candidates]
        candidate_records+= [dict(dataset=dataset,**e) for e in confirmation[dataset]]
        # Training data paths are explicit; none is inferred from confirmation labels.
        tp=ROOT/('Baselines/SARA-main/data/reformatted/QASPER_train_split.jsonl' if dataset=='QASPER' else 'data/week12/hotpot_train_v1.1.json' if dataset=='HotpotQA' else 'data/week12/scifact/claims_train.jsonl')
        training[dataset]=dict(path=str(tp),available=tp.exists(),filter='exclude development/confirmation question hashes and document groups; historical TRAIN reuse is allowed; never reuse historical validation groups')
        report.append(dict(dataset=dataset,official_source_split='TRAIN(project DEV); official validation candidates' if dataset=='QASPER' else 'DEV',historical_used_ids=sorted(used[dataset]),historical_question_hashes=sorted(used_questions[dataset]),historical_paper_or_document_ids=sorted(used_docs[dataset]),development_ids=sorted(ids[dataset]),candidate_confirmation_ids=[e['query_id'] for e in candidates],requested_confirmation=cfg['confirmation_counts'][dataset],available_proposed_confirmation=len(candidates),training_source=training[dataset],overlap_queries=0,history_coverage='Known project QASPER reformatted sources + complete W11 cohort; W0-W10 additional exposure not yet exhaustively certified',confirmation_ready=False,training_ids_status='NOT_PREPARED; full TRAIN/FIT manifest generated and audited before manual training',indexed_corpus_overlap_is_not_label_leakage=True,document_group_policy='QASPER paper; SciFact connected cited-document components; Hotpot normalized question; corpus documents may be shared'))
    obj=dict(protocol_hash=digest(cfg),development=development,confirmation=confirmation,training=training)
    immutable(out/'cohorts.json',obj)
    immutable(out/'protocol_config.json',cfg)
    from .statistics import registry
    immutable(out/'primary_comparisons.json',registry(cfg,llm=True,lora=True))
    freeze=dict(schema='W12_CONFIRMATION_CANDIDATES_NOT_AUTHORIZED_FOR_EVALUATION',protocol_hash=digest(cfg),ids=confirmation,ids_hash=digest(confirmation),git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),timestamp=utc(),history_status='REVIEW_REQUIRED',confirmation_evaluation_enabled=False)
    if not (out/'W12_CONFIRMATION_FREEZE.json').exists():immutable(out/'W12_CONFIRMATION_FREEZE.json',freeze)
    atomic(out/'split_audit.json',dict(datasets=report,sources=sources,QASPER_TEST_opened=False,full_generation_started=False))
    text=['# Week 12 split and leakage audit','Development reuses the explicit W11 pilot and is NOT confirmation. Candidate confirmation identities are reserved before W12 outcomes, but **not certified fresh** until prior-history review. Week 12 never evaluates them. QASPER TEST remains permanently closed. Historical IDs are scoped by dataset; paper/document grouping and question hashes must be checked, not bare numeric IDs alone.']
    for r in report:
        summary={k:v for k,v in r.items() if k not in ('historical_used_ids','historical_question_hashes','historical_paper_or_document_ids','development_ids','candidate_confirmation_ids')}
        summary.update(historical_used_id_count=len(r['historical_used_ids']),historical_document_count=len(r['historical_paper_or_document_ids']),development_queries=len(r['development_ids']))
        text+=['## '+r['dataset'],json.dumps(summary,indent=2)]
    text+=['## Readiness','HotpotQA/SciFact training files may need explicit preparation. QASPER official-validation candidates must pass historical exposure certification and native-adapter qualification before Week 13. Unknown W0-W10 exposure is a blocker, not evidence of freshness. Full identity lists are in protocol/cohorts.json and protocol/split_audit.json. No fresh QASPER confirmation is claimed.']
    (ROOT/'results/week12/W12_SPLIT_AUDIT.md').write_text('\n\n'.join(text)+'\n');return obj

def development_examples(cfg):
    if cfg.get('split_protocol')=='TRAIN_VAL_CONFIRM_v2':
        from .splits import load_examples
        return load_examples(cfg,'VAL')
    from .protocol import cohort
    _,all_examples=w11_examples();ids=cohort(cfg)
    from src.week11.datasets import read_jsonl
    claims=read_jsonl(ROOT/cfg['datasets']['SciFact']['path']);groups=components([dict(query_id=str(r['id']),split_documents=list(map(str,r.get('cited_doc_ids',[])))) for r in claims])
    result=[e for d,x in all_examples.items() for e in x if e['query_id'] in {v['query_id'] for v in ids[d]}]
    for e in result:
        if e['dataset']=='SciFact':e['cluster_id']='cited-paper-component:'+groups[e['query_id']]
        elif e['dataset']=='HotpotQA':e['cluster_id']='question:'+qhash(e['question'])
    return result
