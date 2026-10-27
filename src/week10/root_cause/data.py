"""Full paper candidate corpus, shared by all retrievers. Gold isolated from scoring."""
import json
import re
from .storage import digest

def clean(text): return ' '.join(text.split())

def normalized_offsets(text):
    chars=[]; positions=[]; pending=None
    for i,ch in enumerate(text):
        if ch.isspace():
            if chars: pending=i if pending is None else pending
        else:
            if pending is not None: chars.append(' '); positions.append(pending); pending=None
            chars.append(ch); positions.append(i)
    return ''.join(chars),positions


def evidence_occurrences(text,evidence):
    normalized,positions=normalized_offsets(text)
    result=[]
    for item in evidence:
        needle=clean(re.sub(r'BIBREF\d+','',item)); starts=[];cursor=0
        if needle:
            while True:
                a=normalized.find(needle,cursor)
                if a<0:break
                starts.append([positions[a],positions[a+len(needle)-1]+1]);cursor=a+1
        result.append(starts)
    return result


def prepare(cfg):
    from datasets import Dataset
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    from .protocol import guard
    guard(cfg['split'])
    raw=Dataset.from_file(cfg['raw_dataset'])
    parser=SentenceSplitter(separator='\n',chunk_size=256,chunk_overlap=0)
    rows=[json.loads(line) for line in open(cfg['dataset']) if line.strip()]
    if cfg.get('query_limit'): rows=rows[:cfg['query_limit']]
    if len({str(r['id']) for r in rows})!=len(rows): raise ValueError('Duplicate query identity')
    corpora={}; documents={}; output=[]
    for row in rows:
        pid=str(row['example_id']); paper=raw[int(pid)]
        if pid not in corpora:
            sections=[paper['title']+'\t'+paper['abstract']]
            for name,paras in zip(paper['full_text']['section_name'],paper['full_text']['paragraphs']):
                if paras and paras!=['']: sections.append(re.sub(r'BIBREF\d+','',str(name)+'\t'+' '.join(paras)))
            bases=[];cursor=0
            for section in sections:bases.append(cursor);cursor+=len(section)+2
            documents[pid]='\n\n'.join(sections)
            nodes=parser.get_nodes_from_documents([Document(text=t,id_=str(j)) for j,t in enumerate(sections)])
            chunks=[]
            for i,node in enumerate(nodes):
                base=bases[int(node.ref_doc_id)];a=base+node.start_char_idx;b=base+node.end_char_idx
                if documents[pid][a:b]!=node.get_content():raise ValueError('Chunk source offsets do not match exact text')
                chunks.append(dict(chunk_id=digest([pid,i,node.get_content()]),paper_id=pid,order=i,text=node.get_content(),char_start=a,char_end=b))
            corpora[pid]=chunks
        q=str(row['question']).strip(); matches=[i for i,t in enumerate(paper['qas']['question']) if t.strip()==q]
        if not matches: raise ValueError('Raw question alignment missing')
        evidence=[]
        for i in matches:
            ev=set(e for a in paper['qas']['answers'][i]['answer'] for e in a.get('evidence',[]) if e.strip())
            evidence.append(ev)
        if any(e!=evidence[0] for e in evidence): raise ValueError('Ambiguous raw evidence alignment')
        chunks=corpora[pid]
        gold=sorted(evidence[0]);occurrences=evidence_occurrences(documents[pid],gold)
        answers=row.get('answer',[]); answers=answers if isinstance(answers,list) else [answers]
        reform=row.get('answer_reformatted',answers); reform=reform if isinstance(reform,list) else [reform]
        annotations=[a for i in matches for a in paper['qas']['answers'][i]['answer']]
        output.append(dict(inference=dict(question_id=str(row['id']),paper_id=pid,question=q,chunks=chunks),
            evaluation=dict(answers=answers,answers_reformatted=reform,gold_evidence=gold,gold_occurrences=occurrences,
                question_type=row.get('question_type'),annotations=annotations)))
    return output
