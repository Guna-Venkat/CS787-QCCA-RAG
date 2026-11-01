"""Read-only paired retriever interpretation; never changes scores or freezes."""
from pathlib import Path
import sys,json,itertools
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.week11.protocol import load,gate,output
from src.week11.cache import Store,file_hash,atomic
from src.week11.bootstrap import paired

def main():
    cfg=load();gate(cfg,'G');root=output(cfg,'F')
    manifest=json.loads((root/'manifest.json').read_text());rows=Store(root/'rows',manifest['identity']).rows()
    assert len(rows)==2400
    frame=pd.DataFrame(rows);summary=pd.read_csv(root/'retriever_robustness_summary_df.csv');assert len(summary)==48
    active=frame[frame.evidence_granularity!='NONE'];contrasts=[];identities=[]
    for (dataset,gran,selector,budget),g in active.groupby(['dataset','evidence_granularity','selector','budget']):
        for left,right in itertools.combinations(['BM25','MiniLM','SFR'],2):
            a=g[g.retriever==left].to_dict('records');b=g[g.retriever==right].to_dict('records');assert len(a)==len(b)==50
            result=paired(a,b,n=10000,seed=42)
            contrasts.append(dict(dataset=dataset,granularity=gran,selector=selector,budget=budget,contrast=f'{left}_minus_{right}',left_score=sum(r['task_score'] for r in a)/50,right_score=sum(r['task_score'] for r in b)/50,**result))
            bm={r['query_id']:r for r in b}
            identities.append(dict(dataset=dataset,granularity=gran,selector=selector,budget=budget,comparison=f'{left}_vs_{right}',queries=50,same_candidate_set=sum(set(r['candidate_pool_ids'])==set(bm[r['query_id']]['candidate_pool_ids']) for r in a),same_evidence=sum(r['evidence_hash']==bm[r['query_id']]['evidence_hash'] for r in a),same_prediction=sum(r['prediction']==bm[r['query_id']]['prediction'] for r in a)))
    comparisons=pd.DataFrame(contrasts);identity=pd.DataFrame(identities)
    comparisons.to_csv(root/'named_retriever_comparisons.csv',index=False);identity.to_csv(root/'retriever_identity_diagnostic.csv',index=False)
    assert len(comparisons)==45
    cols=['dataset','retriever','evidence_granularity','selector','budget','task_score','delta_task_score','HELP','HURT','actual_evidence_tokens','prompt_tokens','generated_tokens']
    passage=comparisons[(comparisons.granularity=='PASSAGE')&(comparisons.budget.astype(str)=='1024')]
    hot=identity[identity.dataset=='HotpotQA'];assert hot.same_candidate_set.eq(50).all() and hot.same_evidence.eq(50).all() and hot.same_prediction.eq(50).all()
    lines=['# Week 11 F — retriever robustness: research review',
    '## Completion and protocol',
    'All 2,400 checksummed logical rows are complete: 150 frozen development queries, 50 per dataset, 48 dataset/condition summaries. Qwen is held fixed. Five manually reviewed representation/budget configurations are each evaluated under BM25, MiniLM and SFR, plus one no-evidence baseline per dataset. All policies use the same source corpus and pool-width policy (20), P0 passages, frozen semantic/hybrid downstream selection and decoding. No old result is overwritten.',
    'The five controls are semantic passage/1024, semantic sentence/1024, semantic proposition/1024, hybrid sentence/256 and semantic passage/FULL. They were reviewed before this F run; they are exploratory finalists from the same development pilot, not independently validated winners. FULL is the capped candidate pool and context policy, not unrestricted corpus input.',
    '## Observations',
    '**HotpotQA does not identify a retriever effect in this setup.** All 15 matched pair/configuration contrasts have identical candidate sets, exact evidence hashes and predictions on all 50 queries. Width 20 already includes every provided distractor-context paragraph (10). The downstream semantic/hybrid selector then exposes identical evidence. These are repeated identical interventions, not evidence of general retriever robustness on open-corpus HotpotQA.',
    '**QASPER retriever differences are small on this pilot.** At passage/1024, BM25 F1 is 0.201220, MiniLM 0.197917 and SFR 0.199183. At passage/FULL they are 0.249211, 0.241667 and 0.243652. Candidate-pool differences can affect exposed context, but neither these point estimates nor a zero-containing interval establish equivalence.',
    '**SciFact shows larger point-estimate sensitivity.** At passage/1024, BM25 strict accuracy is 0.76, MiniLM 0.70 and SFR 0.68; the differences are only a few claims out of 50. Compact hybrid sentence/256 gives 0.74, 0.64 and 0.72 respectively. These are task-specific exploratory contrasts, not a universal retriever ranking. Full paired intervals below must govern any superiority claim.',
    'HotpotQA passage/1024 remains 0.480272 F1 for all retrievers, versus no-evidence 0.150250. QASPER no-evidence is 0.091228 F1; SciFact no-evidence is 0.36 strict accuracy. QA F1 and verification accuracy are reported separately, never pooled.',
    '## Matched passage/1024 retriever contrasts',passage.to_markdown(index=False,floatfmt='.6f'),
    '## All 45 matched retriever contrasts',
    'Paired bootstrap uses 10,000 resamples, seed 42; QASPER clusters by paper (40 clusters for 50 queries), other datasets by query/claim (50). These are nominal exploratory 95% intervals without multiple-comparison adjustment. The contrast direction is explicitly named. No inference of superiority is made from point estimates alone.',comparisons.to_markdown(index=False,floatfmt='.6f'),
    'The only contrast whose nominal interval strictly excludes zero is SciFact hybrid sentence/256, MiniLM minus SFR: −0.08, CI [−0.16, −0.02]. This is one exploratory result among 45 unadjusted comparisons. BM25 minus SFR passage/1024 is +0.08 with CI [0.00, 0.18], which does not establish a positive lower bound.',
    '## Candidate/evidence identity diagnostic',identity.to_markdown(index=False),
    'Candidate identity compares sets rather than retrieval rank order. Evidence hashes check the actual generator exposure; prediction equality is separate. Shared predictions from a cache are not independent repeated model experiments.',
    '## Flags and cost',
    'The mechanical audit found zero empty-evidence rows, zero generation-cap hits and zero invalid SciFact labels. There are 112 annotation-mapping flag rows: the same seven QASPER questions across 16 conditions, not 112 distinct failed queries. Gold mapping affects retrospective evidence metrics, not sentence/passage ranking or answer utility scoring. Full identifiers remain in flagged_queries.csv.',
    'There were 519 physical generation calls for 2,400 logical policy/query rows; 1,881 reused compatible cached outputs. Conceptual policy inference cost remains one answer-generation call even on a cache hit. Selection/embedding preprocessing is not free. Final/cumulative prompt tokens count actual LM input, not all corpus text in memory. Cached timing fields exclude preparation/loading/IPC overhead; end-to-end serving latency was not measured and remains null. No wall-clock speed superiority is claimed.',
    '## Continuation and limits',
    'Retain Qwen/SFR for G exactly as planned, rather than silently changing the controlled hierarchy experiment after observing these F scores. F does not establish a universal retriever winner or open-corpus robustness. The 50-per-dataset cohort is a development pilot; stronger claims need larger held-out confirmation. Optional LLMLingua-2 remains pending elsewhere; F itself has no pending executable conditions.',
    '## Complete 48-condition table',summary[cols].sort_values(['dataset','evidence_granularity','selector','budget','retriever'],key=lambda x:x.astype(str)).to_markdown(index=False,floatfmt='.6f')]
    (root/'RESEARCH_REVIEW.md').write_text('\n\n'.join(lines)+'\n')
    atomic(root/'research_review_provenance.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),source_status_sha256=file_hash(root/'status.json'),driver_sha256=file_hash(Path(__file__)),named_paired_contrasts=45,primary_scores_changed=False))
    print(comparisons.to_string(index=False))
if __name__=='__main__':main()
