import re
import string
from collections import Counter
import numpy as np
from .protocol import guard


def hotpot_normalize(text):
    text=''.join(c for c in str(text).lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def qa_pair(prediction,reference):
    p,g=hotpot_normalize(prediction),hotpot_normalize(reference)
    if p!=g and (p in ('yes','no','noanswer') or g in ('yes','no','noanswer')):return 0.,0.
    a,b=p.split(),g.split();overlap=sum((Counter(a)&Counter(b)).values())
    f1=2*overlap/(len(a)+len(b)) if a or b else 0.
    return f1,float(p==g)


def evaluate(prediction,example):
    guard(example['dataset'],example['split']);ev=example['evaluation']
    if example['task_type']=='verification':
        label=prediction.strip().upper()
        correct=float(label==ev['gold_label'])
        return dict(task_score=correct,classification_accuracy=correct,predicted_label=label,
                    f1=None,exact_match=None,metric='classification_accuracy')
    if example['dataset']=='QASPER':
        from src.week6.evaluation import compute_qa_answer_metrics
        score=compute_qa_answer_metrics(prediction,ev['gold_answer'],ev.get('answers_reformatted'))
        f1,em=score['token_f1'],score['exact_match']
    else:
        scores=[qa_pair(prediction,answer) for answer in ev['gold_answer'] or ['']]
        f1=max(s[0] for s in scores);em=max(s[1] for s in scores)
    return dict(task_score=float(f1),f1=float(f1),exact_match=float(em),classification_accuracy=None,metric='answer_f1')


def help_hurt(score,baseline,classification=False,epsilon=.01):
    delta=score-baseline
    return dict(delta_task_score=delta,HELP=delta>(0 if classification else epsilon),
                HURT=delta<-(0 if classification else epsilon))


def classification_metrics(gold,predicted):
    labels=['SUPPORT','REFUTE','NOT_ENOUGH_INFO'];result={};values=[]
    for label in labels:
        tp=sum(g==label and p==label for g,p in zip(gold,predicted))
        fp=sum(g!=label and p==label for g,p in zip(gold,predicted))
        fn=sum(g==label and p!=label for g,p in zip(gold,predicted))
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;values.append(f1)
        result[label]=dict(precision=precision,recall=recall,f1=f1,support=sum(g==label for g in gold))
    return dict(accuracy=float(np.mean([g==p for g,p in zip(gold,predicted)])),macro_f1=float(np.mean(values)),classes=result)


def span_coverage(a,b,units,text=None):
    cursor=a;covered=0
    for u in sorted(units,key=lambda u:u['start_char']):
        x,y=max(a,u['start_char']),min(b,u['end_char'])
        if y>max(cursor,x):
            start=max(cursor,x)
            covered+=sum(not c.isspace() for c in text[start:y]) if text is not None else y-start
            cursor=y
    denominator=sum(not c.isspace() for c in text[a:b]) if text is not None else b-a
    return covered/max(1,denominator)


def evidence_metrics(selected,example):
    gold=example['evaluation']['gold_spans'];coverage=[]
    documents={d['document_id']:d['text'] for d in example['inference']['documents']}
    for span in gold:
        units=[u for u in selected if u['document_id']==span['document_id']]
        alternatives=span.get('alternatives') or ([] if span.get('unmapped') else [[span['start_char'],span['end_char']]])
        coverage.append(max([span_coverage(a,b,units,documents.get(span['document_id'])) for a,b in alternatives] or [0.]))
    return dict(any_gold_recall=float(any(x>=1-1e-9 for x in coverage)) if gold else None,
        all_gold_recall=float(all(x>=1-1e-9 for x in coverage)) if gold else None,
        evidence_coverage=float(np.mean(coverage)) if gold else None,
        gold_annotation_count=len(gold),unmapped_gold=sum(g.get('unmapped',False) for g in gold),
        mapping_note='Full non-whitespace annotated-character union coverage; repeated QASPER occurrences are alternatives; no-gold recall is null')


def retrieval_metrics(ranked,example,ks=(1,2,4,6,8,10)):
    first=next((i+1 for i in range(len(ranked)) if evidence_metrics(ranked[:i+1],example)['any_gold_recall']),None)
    result=[]
    for k in ks:
        units=ranked[:k];words=[set(u['source_text'].lower().split()) for u in units]
        pairs=[len(a&b)/max(1,len(a|b)) for i,a in enumerate(words) for b in words[i+1:]]
        metrics=evidence_metrics(units,example)
        result.append(dict(k=k,actual_k=len(units),**metrics,
            first_relevant_rank=first,mrr=1/first if first else 0. if metrics['gold_annotation_count'] else None,
            retrieved_tokens=sum(u['token_count'] for u in units),
            duplicate_rate=1-len({u['hash'] for u in units})/len(units) if units else None,
            redundancy=float(np.mean(pairs)) if pairs else None,candidate_diversity=1-float(np.mean(pairs)) if pairs else None))
    return result
