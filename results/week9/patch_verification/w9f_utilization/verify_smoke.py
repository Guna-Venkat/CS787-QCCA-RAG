"""Read-only audit of real W9-F smoke and historical artifact preservation."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from src.week9.provenance import branch_guard,digest,file_hash
from src.week9.caching import read_jsonl,atomic_json
from src.week9.utilization_f.prompts import definitions,POLICIES
from src.week9.utilization_f.protocol import verify_checksums
from src.week9.utilization_f.runner import invalid_answer


def main():
    branch_guard()
    root=ROOT/'results/week9/w9f_utilization/v1'
    protocol=json.loads((root/'protocol.json').read_text())
    rows=read_jsonl(root/'smoke/results.jsonl')
    calls=read_jsonl(root/'smoke/lm_calls.jsonl')
    ident=dict(protocol=protocol,stage='smoke')
    verify_checksums(rows,ident);verify_checksums(calls,ident)
    assert len(rows)==10 and len(calls)==6
    assert json.loads((root/'prompts.json').read_text())==definitions()
    grouped={q:[r for r in rows if r['question_id']==q] for q in {r['question_id'] for r in rows}}
    assert len(grouped)==2
    inspections=[]
    for q,group in sorted(grouped.items()):
        assert {r['policy'] for r in group}==set(('D0',)+POLICIES)
        f0=next(r for r in group if r['policy']=='F0')
        assert len({r['evidence_hash'] for r in group if r['policy']!='D0'})==1
        assert len({digest(r['selected_sentence_ids']) for r in group if r['policy']!='D0'})==1
        for r in group:
            if r['policy']!='D0':
                assert r['evidence_text']==f0['evidence_text']
                assert r['sentence_text_hashes']==[s['text_hash'] for s in f0['selected_sentences']]
            if r['policy'] in ('F1','F2-A','F2-B'):
                assert not invalid_answer(r['raw_generation']) and not r['answer_fail_closed']
                assert all(c['stage']=='revision' for c in r['model_calls'])
                assert r['extraction_prompt_tokens']==r['extraction_generated_tokens']==0
                assert r['num_lm_calls']==2
                assert r['final_prompt_tokens']>f0['final_prompt_tokens']
            else:assert r['new_answer_calls']==0
        inspections.append(dict(question_id=q,paper_id=f0['paper_id'],question=f0['question'],
            initial_draft=f0['initial_draft'],source_chunk_ids=f0['selected_chunk_ids'],
            selected_sentence_ids=f0['selected_sentence_ids'],evidence_hash=f0['evidence_hash'],
            exact_evidence=f0['evidence_text'],evidence_tokens=f0['evidence_tokens'],
            answers={r['policy']:dict(prediction=r['prediction'],f1=r['f1'],final_prompt_tokens=r['final_prompt_tokens']) for r in group}))
    for p,h in protocol['input_hashes'].items():assert file_hash(p)==h,p
    proof=json.loads((root/'f0_alias_verification.json').read_text())
    assert proof['equivalent'] and proof['queries']==1746 and all(r['equivalent'] for r in proof['proofs'])
    assert not (root/'fit').exists()
    for name in ('frozen_w9f_f2.json','frozen_w9f_intervention.json','w9f_controller_handoff.json'):
        assert not (root/name).exists()
    peak=max(c.get('peak_memory_allocated',0) for r in rows if r['policy'] in ('F1','F2-A','F2-B') for c in r['model_calls'])/1024**3
    report=dict(branch='gagan',full_w9f_started=False,controller_trained_or_run=False,
        f0_alias_queries_verified=1746,smoke_queries=2,smoke_rows=10,new_revision_calls=6,
        extraction_lm_calls=0,evidence_identical=True,all_source_artifacts_unchanged=True,
        no_wrapper_or_reasoning_outputs=True,peak_allocated_gpu_gib=peak,
        protocol_hash=digest(protocol),inspections=inspections,
        interpretation='Mechanics only. No prompt/mode is selected from smoke. F2-B changing an answer does not prove evidence support or general correction ability.')
    atomic_json(Path(__file__).with_name('verification.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='inspections'},indent=2))


if __name__=='__main__':main()
