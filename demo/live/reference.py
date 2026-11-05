"""Post-generation demo-reference diagnostics, never inference inputs or research scores."""
from collections import Counter
import re
from .preset import is_preset,load_preset


def checklist(answer):
    """Conservative surface-cue matching; not entailment or contradiction detection."""
    text=answer.lower().replace('’',"'")
    text=re.sub(r'\b(one|two|three|four|seven|eight|twelve|sixteen)\b',
                lambda m:dict(one='1',two='2',three='3',four='4',seven='7',
                              eight='8',twelve='12',sixteen='16')[m[0]],text)
    text=re.sub(r'[-–—→]',' ',text)
    text=re.sub(r'\s+',' ',text)
    # Each fact's cues must co-occur within one sentence/semicolon clause.
    clauses=re.split(r'[.!?;]',text)
    return [dict(Fact=f['text'],Matched=any(all(re.search(p,c) for p in f['patterns'])
                                          for c in clauses))
            for f in load_preset()['reference']['required_facts']]


def evaluate(result):
    if not is_preset(result['query'],result['passages']) or result['answer'] is None:
        return None
    from src.week11.evaluation import hotpot_normalize,qa_pair
    answer=result['answer'];ref=load_preset()['reference']['answer']
    a,b=hotpot_normalize(answer).split(),hotpot_normalize(ref).split()
    overlap=sum((Counter(a)&Counter(b)).values())
    f1,em=qa_pair(answer,ref)
    facts=checklist(answer);matched=sum(f['Matched'] for f in facts)
    return dict(precision=overlap/len(a) if a else 0.,
                recall=overlap/len(b) if b else 0.,f1=f1,em=em,
                matched_facts=matched,total_facts=len(facts),
                fact_cue_coverage=matched/len(facts),checklist=facts)


def metrics_text(result):
    if not is_preset(result['query'],result['passages']):
        return 'No gold-reference evaluation available'
    m=evaluate(result)
    if m is None:return 'Not evaluated — no generated answer'
    return (f"F1 {m['f1']:.3f} · EM {m['em']:.0f} · "
            f"P {m['precision']:.3f} · R {m['recall']:.3f} · "
            f"Fact cues {m['matched_facts']}/{m['total_facts']} "
            f"({m['fact_cue_coverage']:.0%})")
