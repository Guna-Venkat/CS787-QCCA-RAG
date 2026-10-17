"""Repair only whitespace-affected masking queries; preserve valid cached semantics."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from tqdm.auto import tqdm
from src.week8.data import load_config,load_week8_split
from src.week8.generator_features import (source_fingerprint,canonical_k4_context,
    compute_query_features,validate_feature_rows,FEATURE_COLUMNS)
from src.week8.utility_training import load_surface
from src.week8.gate_diagnostics import measure_draft_batch,gate_source_fingerprint,summarize_diagnostics
from src.week8.semantic import SemanticScorer


def main():
    torch.set_num_threads(4)
    root=ROOT/'results/week8';archive=root/'archive_w8d_question_normalization'
    plan=json.loads((archive/'repair_plan.json').read_text());cfg=load_config()
    import src
    base=str(ROOT/'Baselines/SARA-main/src')
    if base not in src.__path__:src.__path__.append(base)
    from src.model.loader import load_sara_for_eval
    if torch.cuda.mem_get_info()[0]<18*1024**3:raise RuntimeError('Need 18 GiB free GPU memory for targeted repair')
    model,tokenizer=load_sara_for_eval(str(ROOT/cfg['sara']['checkpoint_path']),device='cuda')
    scorer=SemanticScorer(device='cpu');started=time.perf_counter();repaired={}
    try:
        for split in ['fit','internal_val']:
            df=load_surface(split);groups={str(q):g for q,g in df.groupby('question_id')}
            records,sfr,_=load_week8_split(split);records={str(r['question_id']):r for r in records}
            fp,payload=source_fingerprint(split)
            if fp!=plan['splits'][split]['new_fingerprint']:raise ValueError('Repair implementation changed from reviewed plan')
            path=root/'w8d_attribution/full'/f'features_{split}.jsonl'
            entries={str(e['question_id']):e for e in map(json.loads,path.read_text().splitlines())}
            oldpath=archive/'attribution_v1/full'/f'features_{split}.jsonl'
            old={str(e['question_id']):e for e in map(json.loads,oldpath.read_text().splitlines())}
            targets=plan['splits'][split]['affected_queries']
            if split=='fit' and '1816' not in entries:
                q='1816';g=groups[q]
                question,prompt,_,_,embeds,_=canonical_k4_context(g,records[q],sfr[q],tokenizer)
                measured=measure_draft_batch(model,tokenizer,[{'question_id':q,'prompt':prompt,
                    'draft_answer':str(g.iloc[0].draft_answer),'embeds':embeds}],'cuda')[q]
                semantic=float(scorer.score_query_passages(question,[str(g.iloc[0].draft_answer)]).cpu()[0])
                features=summarize_diagnostics(measured,question,str(g.iloc[0].draft_answer),semantic)
                gatefp,_,_,_=gate_source_fingerprint('fit')
                row={'question_id':q,'paper_id':str(g.iloc[0].paper_id),'question':question,
                    'draft_answer':str(g.iloc[0].draft_answer),'source_fingerprint':gatefp,'features':features,**measured}
                gatepath=root/'w8e_gate_inputs/canonical_diagnostics_fit.jsonl'
                existing={str(json.loads(l)['question_id']) for l in gatepath.read_text().splitlines()}
                if q not in existing:
                    with gatepath.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print('Query 1816 canonical generation reproduced after question normalization',flush=True)
            done=[]
            with path.open('a') as f:
                for q in tqdm([q for q in targets if q not in entries],desc=f'W8-D targeted repair {split}',unit='query'):
                    rows=compute_query_features(groups[q],records[q],sfr[q],model,tokenizer,scorer,fp,'cuda')
                    oldrows={int(r['semantic_rank']):r for r in old[q]['candidates']}
                    for row in rows:
                        previous=oldrows[int(row['semantic_rank'])]
                        # The question normalization changes only the LLM prompt,
                        # not the already valid MiniLM semantic feature definition.
                        for name in ['query_relevance_rq','draft_relevance_ry','redundancy_red',
                                     'candidate_diversity','mig_score','a6_draft_sim']:
                            if not np.isclose(row[name],previous[name],atol=1e-6,rtol=0):
                                raise ValueError(f'{q}: unexpected semantic feature change')
                            row[name]=previous[name]
                        row['previous_source_fingerprint']=previous['source_fingerprint']
                    validate_feature_rows(rows,groups[q],fp)
                    entry={'question_id':q,'candidates':rows}
                    f.write(json.dumps(entry,allow_nan=False)+'\n');f.flush();entries[q]=entry;done.append(q)
            if set(entries)!=set(groups):raise ValueError('Repair did not restore full split coverage')
            result=pd.DataFrame([r for q in sorted(groups) for r in entries[q]['candidates']])
            result.to_csv(path.with_suffix('.csv'),index=False)
            metadata={**payload,'source_fingerprint':fp,'num_queries':len(groups),'num_rows':len(result),
                'feature_columns':FEATURE_COLUMNS,'attention_A1_A5':'not measured; masking fallback only',
                'question_normalization':'str.strip, identical to W7/W8-A',
                'repaired_query_ids':targets,'semantic_components':'Valid v1 MiniLM components preserved; only generator prompt corrected'}
            (path.parent/f'provenance_{split}.json').write_text(json.dumps(metadata,indent=2))
            repaired[split]=done
    finally:
        del model,tokenizer
        torch.cuda.empty_cache()
    (archive/'repair_completed.json').write_text(json.dumps({'repaired':repaired,'wall_seconds':time.perf_counter()-started,
        'canonical_answers_unchanged':True,'checkpoints_unchanged':True},indent=2))
    print('Targeted question normalization repair completed',flush=True)

if __name__=='__main__':main()
