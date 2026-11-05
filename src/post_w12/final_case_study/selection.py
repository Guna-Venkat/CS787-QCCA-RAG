"""Fixed deterministic explanatory case selection, never an estimate of prevalence."""
from itertools import combinations

METHODS=['FIXED_SENTENCE','QBGF_FULL','BEA_QBGF_V1','BEA_V2']
PREFIXES=['FIXED','QBGF','V1','V2']
PAIRS=list(combinations(PREFIXES,2))
RULES={
 'maximum_disagreement':'largest per-query score spread; for SciFact @256 prefer FIXED-only correct and @1024 V1-only correct among equal spreads; then ascending query ID',
 'v2_rescue':'positive V2−V1 first, largest delta, then number of other methods beating V1, largest spread, query ID; prefer unused query among positive candidates; if none positive explicitly mark no positive rescue',
 'near_tie':'spread <=0.01, max score >0, with token range >=10% budget or minimum pairwise underlying-sentence Jaccard <=0.5; smallest spread, largest allocation difference, highest max score, query ID; prefer unused eligible query; absent eligibility is explicitly flagged',
 'budget_reversal':'strict pairwise method-order inversions with both absolute QA margins >=0.01 (SciFact correctness margins >=1), then largest change in pairwise score margin, then ascending query ID; one query/dataset only if an inversion qualifies',
 'score':'QA per-query F1; SciFact binary correctness, never per-query macro-F1',
 'selection_bias':'up to 18 primary case slots plus 3 budget reversals; duplicate query/budget cases are disclosed and frequencies also use unique query/budget units'
}

def choose(records,datasets):
    chosen=[];reversals=[]
    for d in datasets:
        for b in [256,1024]:
            rs=[r for r in records if r['dataset']==d and r['budget']==b]
            def priority(r):
                wanted='FIXED' if b==256 else 'V1'
                return int(d=='SciFact' and r[wanted+'_score']==1 and
                    all(r[p+'_score']==0 for p in PREFIXES if p!=wanted))
            first=min(rs,key=lambda r:(-r['score_spread'],-priority(r),r['query_id']))
            chosen.append(dict(dataset=d,budget=b,query_id=first['query_id'],case_type='MAXIMUM_DISAGREEMENT',
                               qualifies=first['score_spread']>0,note='Largest score spread; ties follow saved rule'))
            used={first['query_id']}
            positive=[r for r in rs if r['V2_score']>r['V1_score']]
            candidates=[r for r in positive if r['query_id'] not in used] or positive or rs
            rescue=min(candidates,key=lambda r:(-(r['V2_score']-r['V1_score']),
                -sum(r[p+'_score']>r['V1_score'] for p in ['FIXED','QBGF']),-r['score_spread'],r['query_id']))
            chosen.append(dict(dataset=d,budget=b,query_id=rescue['query_id'],case_type='V2_RESCUE',
                qualifies=rescue['V2_score']>rescue['V1_score'],note='No positive rescue exists' if not positive else 'Positive V2−V1, deterministic rank'))
            used.add(rescue['query_id'])
            eligible=[r for r in rs if r['score_spread']<=.01 and r['max_score']>0 and r['meaningfully_different_allocation']]
            candidates=[r for r in eligible if r['query_id'] not in used] or eligible
            if not candidates:
                candidates=[r for r in rs if r['meaningfully_different_allocation'] and r['query_id'] not in used] or rs
            tie=min(candidates,key=lambda r:(r['score_spread'],-r['allocation_difference'],-r['max_score'],r['query_id']))
            chosen.append(dict(dataset=d,budget=b,query_id=tie['query_id'],case_type='NEAR_TIE',
                qualifies=tie in eligible,note='Near tie with meaningful allocation difference' if tie in eligible else 'No qualifying nontrivial near tie; descriptive fallback'))
        by={(r['query_id'],r['budget']):r for r in records if r['dataset']==d};candidates=[]
        for q in sorted({q for q,b in by}):
            a,b=by[q,256],by[q,1024];inversions=[];changes=[]
            for x,y in PAIRS:
                lo=a[x+'_score']-a[y+'_score'];hi=b[x+'_score']-b[y+'_score']
                floor=1 if d=='SciFact' else .01
                if lo*hi<0 and min(abs(lo),abs(hi))>=floor:inversions.append(x+' vs '+y)
                changes.append(abs(hi-lo))
            candidates.append(dict(dataset=d,query_id=q,strict_inversions=len(inversions),
                inverted_pairs='; '.join(inversions),maximum_margin_change=max(changes),
                **{p+'_score_256':a[p+'_score'] for p in PREFIXES},
                **{p+'_score_1024':b[p+'_score'] for p in PREFIXES}))
        if any(r['strict_inversions'] for r in candidates):
            best=min(candidates,key=lambda r:(-r['strict_inversions'],-r['maximum_margin_change'],r['query_id']))
            reversals.append(dict(best,qualifies=True))
            chosen.append(dict(dataset=d,budget='256_vs_1024',query_id=best['query_id'],
                case_type='BUDGET_REVERSAL',qualifies=True,note=best['inverted_pairs']))
        else:reversals.append(dict(dataset=d,query_id=None,strict_inversions=0,qualifies=False,note='No strict ordering inversion; no example fabricated'))
    for i,r in enumerate(chosen,1):
        r['case_id']=f'C{i:02d}'
        r['duplicates_previous_query_budget']=any((x['dataset'],x['query_id'],x['budget'])==(r['dataset'],r['query_id'],r['budget']) for x in chosen[:i-1])
    return chosen,reversals
