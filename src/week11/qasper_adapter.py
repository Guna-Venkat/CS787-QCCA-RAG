import json
import re
from datasets import Dataset
from src.week10.root_cause.data import evidence_occurrences
from .protocol import guard
from .datasets import normalize, document


def load(spec):
    guard('QASPER',spec['split'])
    from pathlib import Path
    if any('test' in Path(spec[k]).name.lower() for k in ('path','raw_path')):
        raise ValueError('QASPER TEST inputs blocked before reading')
    raw=Dataset.from_file(spec['raw_path']);examples=[];docs={}
    for line in open(spec['path']):
        if not line.strip():continue
        row=json.loads(line);pid=str(row['example_id']);paper=raw[int(pid)]
        if pid not in docs:
            sections=[paper['title']+'\t'+paper['abstract']]
            for name,paras in zip(paper['full_text']['section_name'],paper['full_text']['paragraphs']):
                if paras and paras!=['']:sections.append(re.sub(r'BIBREF\d+','',str(name)+'\t'+' '.join(paras)))
            # Native sections can be very large: deterministic bounded passage
            # chunker is used for scientific-paper passages, as frozen in config.
            docs[pid]=document('QASPER',pid,'\n\n'.join(sections))
        q=str(row['question']).strip()
        matches=[i for i,t in enumerate(paper['qas']['question']) if t.strip()==q]
        if not matches:raise ValueError('QASPER raw question alignment missing')
        sets=[set(e for a in paper['qas']['answers'][i]['answer'] for e in a.get('evidence',[]) if e.strip()) for i in matches]
        if any(x!=sets[0] for x in sets):raise ValueError('Ambiguous QASPER evidence alignment')
        gold=[];texts=sorted(sets[0]);occ=evidence_occurrences(docs[pid]['text'],texts)
        for index,(text,alternatives) in enumerate(zip(texts,occ)):
            gold.append(dict(document_id=pid,evidence_id=f'{pid}:gold{index}',text=text,
                alternatives=alternatives,start_char=alternatives[0][0] if alternatives else None,
                end_char=alternatives[0][1] if alternatives else None,unmapped=not bool(alternatives)))
        answers=row.get('answer',[]);answers=answers if isinstance(answers,list) else [answers]
        example=normalize('QASPER',spec['split'],row['id'],q,[docs[pid]],answers,gold=gold,
            answer_type=row.get('question_type','unknown'),metadata=dict(corpus='full matching paper',cluster='paper'))
        example['evaluation']['answers_reformatted']=row.get('answer_reformatted',answers)
        examples.append(example)
    return examples
