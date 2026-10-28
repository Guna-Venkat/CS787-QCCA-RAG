"""Read-only hierarchy interpretation and named paired controls."""
from pathlib import Path
import sys,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.week11.protocol import load,gate,output
from src.week11.cache import Store,file_hash,atomic
from src.week11.bootstrap import paired

def stored(root):
    manifest=json.loads((root/'manifest.json').read_text())
    return pd.DataFrame(Store(root/'rows',manifest['identity']).rows())

def main():
    cfg=load();gate(cfg,'H');root=output(cfg,'G');frame=stored(root);assert len(frame)==3150
    control=stored(output(cfg,'C'));assert len(control)==5550
    summary=pd.read_csv(root/'hierarchy_summary_df.csv')
    named=[];fixed=[];reuse=[]
    for dataset in cfg['datasets']:
        for budget in cfg['budgets']:
            g=frame[(frame.dataset==dataset)&(frame.budget.astype(str)==str(budget))]
            h0=g[g.strategy=='H0'].to_dict('records');assert len(h0)==50
            for gran in ['PASSAGE','SENTENCE','PROPOSITION']:
                c=control[(control.dataset==dataset)&(control.budget.astype(str)==str(budget))&(control.evidence_granularity==gran)&(control.selector=='semantic')].to_dict('records');assert len(c)==50
                if gran=='PASSAGE':
                    byid={r['query_id']:r for r in c}
                    reuse.append(dict(dataset=dataset,budget=budget,queries=50,same_evidence=sum(r['evidence_hash']==byid[r['query_id']]['evidence_hash'] for r in h0),same_prediction=sum(r['prediction']==byid[r['query_id']]['prediction'] for r in h0)))
                for strategy in ['H1','H2','H3','H5']:
                    a=g[g.strategy==strategy].to_dict('records');assert len(a)==50
                    if gran=='PASSAGE':
                        named.append(dict(dataset=dataset,budget=budget,strategy=strategy,contrast=strategy+'_minus_H0',candidate_score=sum(r['task_score'] for r in a)/50,control_score=sum(r['task_score'] for r in h0)/50,**paired(a,h0,n=10000,seed=42)))
                    else:
                        fixed.append(dict(dataset=dataset,budget=budget,strategy=strategy,contrast=strategy+'_minus_fixed_'+gran,candidate_score=sum(r['task_score'] for r in a)/50,control_score=sum(r['task_score'] for r in c)/50,**paired(a,c,n=10000,seed=42)))
    named=pd.DataFrame(named);fixed=pd.DataFrame(fixed);reuse=pd.DataFrame(reuse)
    named.to_csv(root/'named_hierarchy_vs_H0.csv',index=False);fixed.to_csv(root/'named_hierarchy_vs_fixed_granularity.csv',index=False);reuse.to_csv(root/'H0_fixed_passage_identity.csv',index=False)
    frame['empty']=(frame.evidence_granularity!='NONE')&frame.selected_unit_ids.map(len).eq(0)
    frame['cap']=frame.generated_tokens.ge(128)
    frame['invalid']=(frame.dataset=='SciFact')&(~frame.prediction.str.strip().str.upper().isin(['SUPPORT','REFUTE','NOT_ENOUGH_INFO']))
    flags=frame.groupby(['dataset','strategy']).agg(empty=('empty','sum'),cap=('cap','sum'),invalid=('invalid','sum')).reset_index();flags.to_csv(root/'flags_by_strategy.csv',index=False)
    cols=['dataset','strategy','budget','task_score','delta_task_score','HELP','HURT','actual_evidence_tokens','prompt_tokens','generated_tokens']
    peaks=summary[summary.evidence_granularity!='NONE'].sort_values(['task_score','prompt_tokens'],ascending=[False,True]).groupby(['dataset','strategy'],sort=True).head(1)[cols]
    oracle=pd.read_csv(root/'oracle_summary_df.csv')
    positive=named[named.ci_low>0];negative=named[named.ci_high<0]
    text=['# Week 11 G — deterministic hierarchical exposure: research review',
    '## Completion and controlled experiment',
    'The available surface contains 3,150 checksummed rows: the same 150 development queries, Qwen/SFR/P0, top-20 capped passage pools, one no-evidence baseline and five available strategies at 256/512/1024/FULL. H4 (LLMLingua-2 tail compression) remains pending: 600 configured rows. This is a frozen deterministic hierarchy prototype, not a trained dynamic controller.',
    '| Strategy | Mechanism |\n|---|---|\n| H0 | SFR-ranked whole passages, strict budget prefix |\n| H1 | First ranked passage in full, then semantic sentences from remaining passages |\n| H2 | First ranked passage in full, then semantic exact-clause propositions from remaining passages |\n| H3 | Semantic sentences from first three ranked passages, then semantic propositions from remaining passages |\n| H4 | Full head + sentence middle + token-compressed tail; pending, no results |\n| H5 | Token-density ranking over first full passage plus all candidate sentences/propositions, gamma=1 |',
    'All available exposures are exact source spans, overlap is suppressed, and selected spans are presented in original source order. Numeric budgets use the shared maximum across the three pinned reader tokenizers. Strict allocation stops at the first oversized unit, so H1/H2 can inherit the same empty-context failure as H0 when their first full passage exceeds budget. FULL remains capped; propositions are deterministic clauses, not generated facts. No outcome or gold label enters selection.',
    '## Best observed point per strategy',
    'These budget peaks are retrospective within the same 50-per-dataset pilot. QA utility is F1; SciFact is strict label accuracy. They are descriptive and do not validate a budget selector.',peaks.sort_values(['dataset','strategy']).to_markdown(index=False,floatfmt='.6f'),
    '## Interpretation',
    'H3 at 256 requested tokens improves over raw H0 on all three datasets: QASPER +0.085380 F1, CI [0.031609, 0.143167]; HotpotQA +0.099869 F1, CI [0.025947, 0.180600]; SciFact +0.32 accuracy, CI [0.14, 0.50]. These are the only three hierarchy-minus-H0 intervals with strictly positive lower bounds. The contrast includes avoiding oversized first passages and exposing smaller units; it is not a clean causal demonstration that mixing granularities itself is necessary.',
    'Against fixed semantic sentences at 256, H3 differences are QASPER −0.028979 (CI [−0.082065, 0.024135]), HotpotQA +0.016114 ([−0.062096, 0.093277]) and SciFact +0.02 ([−0.10, 0.14]). All include zero. Thus the supported low-budget finding is compact source exposure over raw strict-prefix passages; a hierarchy advantage over simple sentence selection is not established.',
    'At 1024, H3 versus H0 is QASPER −0.017076 (CI [−0.050733, 0.014475]), HotpotQA +0.006897 ([−0.051911, 0.068590]) and SciFact −0.02 ([−0.06, 0.00]). The HotpotQA best point estimate of 0.487169 therefore does not establish superiority over raw 0.480272. QASPER FULL strategies converge around 0.243–0.246, also offering no demonstrated hierarchy breakthrough.',
    'H0 exactly reproduces all 600 corresponding C semantic-passage evidence blocks and predictions, confirming the raw control. The oracle headroom is substantial but retrospective: it motivates carefully tested policy research, not a claim that adaptive control already works.',
    '## Matched hierarchy versus H0',
    'Each row pairs the same queries, budget, reader and retriever. Bootstrap: 10,000 resamples, seed 42; QASPER paper clusters (40), others query/claim clusters (50). All intervals are nominal exploratory 95% intervals without correction for multiple comparisons.',named.to_markdown(index=False,floatfmt='.6f'),
    f'{len(positive)} of 48 hierarchy-minus-H0 intervals have a strictly positive lower bound; {len(negative)} have a strictly negative upper bound. These counts are exploratory, not multiplicity-adjusted discoveries.',
    '## Fixed-granularity controls from C',
    'These controls hold Qwen, SFR/P0, query, budget and candidate-pool policy fixed and compare against semantic sentence/proposition allocation. They distinguish a hierarchy effect from merely exposing smaller source units. The additional 96 intervals are exploratory.',fixed.to_markdown(index=False,floatfmt='.6f'),
    '## H0 versus C semantic passage identity',reuse.to_markdown(index=False),
    'This identity diagnostic checks evidence and predictions separately. H0 follows retrieval ordering, whereas C explicitly ranks passages with the selector; the table shows whether they realize the same intervention. Exact duplicated outputs are not independent repeated trials.',
    '## Retrospective oracle',oracle.to_markdown(index=False,floatfmt='.6f'),
    'These ceilings choose actions after seeing the true per-query outcomes. They do not demonstrate a deployable controller, causal evidence discovery, or a successful stopping policy. The oracle includes the available G action surface only; unqualified H4 is absent. Best-fixed scores and ceiling gaps are sensitive to the number of actions tried on this small pilot. H consolidates the broader saved action surface separately.',
    '## Flags',flags.to_markdown(index=False),
    f"Total flags derived here: {int(frame['empty'].sum())} empty-evidence rows, {int(frame.cap.sum())} generation-cap rows, {int(frame.invalid.sum())} invalid SciFact labels. The frozen annotation-mapping flags persist for the same seven QASPER queries; see flagged_queries.csv for identifiers. No flagged row is dropped, reparsed or regenerated.",
    '## Cost and scope',
    f"{int(frame.physical_lm_calls.sum())} physical generation calls served 3,150 logical rows; compatible cache reuse does not make conceptual inference cost zero. No extraction LM or learned controller is added. Embedding preprocessing and ranking have nonzero conceptual cost. Prompt tokens count actual LM input. End-to-end latency was not measured; cached generation/selection fields exclude some loading, preparation and IPC overhead and cannot establish serving speed superiority.",
    'The hypothesis is whether deterministic heterogeneous source exposure improves the utility/token tradeoff under fixed reader/retriever controls. Mixed or zero-containing intervals do not establish equivalence. Any apparent winner still needs a larger preregistered held-out study. No automatic new policy or reader selection is made; H/I are saved-result analyses only.',
    '## Complete available condition table',summary[cols].sort_values(['dataset','strategy','budget'],key=lambda x:x.astype(str)).to_markdown(index=False,floatfmt='.6f')]
    (root/'RESEARCH_REVIEW.md').write_text('\n\n'.join(text)+'\n')
    atomic(root/'research_review_provenance.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),source_status_sha256=file_hash(root/'status.json'),driver_sha256=file_hash(Path(__file__)),hierarchy_comparisons=48,fixed_granularity_comparisons=96,primary_scores_changed=False))
    print(peaks.to_string(index=False));print('Positive H0 contrasts:');print(positive.to_string(index=False));print(oracle.to_string(index=False))
if __name__=='__main__':main()
