"""Supplement the frozen D results with named, exploratory paired analyses."""
from pathlib import Path
import sys
import json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.week11.protocol import load,gate,output
from src.week11.cache import Store,file_hash,atomic
from src.week11.bootstrap import paired
from src.week11.factorial import difference_in_differences


def main():
    cfg=load();gate(cfg,'E');root=output(cfg,'D')
    manifest=json.loads((root/'manifest.json').read_text())
    rows=Store(root/'rows',manifest['identity']).rows();assert len(rows)==16650
    frame=pd.DataFrame(rows);summary=pd.read_csv(root/'reader_granularity_summary_df.csv')
    assert len(summary)==333 and summary.n_queries.eq(50).all()
    def condition(dataset,reader,granularity,budget):
        g=frame[(frame.dataset==dataset)&(frame.reader==reader)&(frame.evidence_granularity==granularity)&(frame.selector=='semantic')&(frame.budget.astype(str)==str(budget))]
        assert len(g)==50
        return g.to_dict('records')
    comparisons=[];unit_comparisons=[];interactions=[]
    for dataset in sorted(frame.dataset.unique()):
        for granularity in ['PASSAGE','SENTENCE']:
            for budget in [1024,'FULL']:
                for left,right in [('Qwen','Mistral'),('Llama','Mistral'),('Llama','Qwen')]:
                    result=paired(condition(dataset,left,granularity,budget),condition(dataset,right,granularity,budget),n=10000,seed=42)
                    comparisons.append(dict(dataset=dataset,granularity=granularity,budget=budget,reader_left=left,reader_right=right,family='additional_posthoc_exploratory',**result))
        for reader in cfg['readers']:
            for budget in [1024,'FULL']:
                a=condition(dataset,reader,'SENTENCE',budget);b=condition(dataset,reader,'PASSAGE',budget)
                result=paired(a,b,n=10000,seed=42)
                unit_comparisons.append(dict(dataset=dataset,reader=reader,budget=budget,contrast='sentence_minus_passage',sentence_tokens=sum(r['actual_evidence_tokens'] for r in a)/50,passage_tokens=sum(r['actual_evidence_tokens'] for r in b)/50,family='additional_posthoc_exploratory',**result))
        if dataset!='SciFact':
            for reader in ['Mistral','Llama']:
                for budget in [1024,'FULL']:
                    cells=[condition(dataset,r,g,budget) for r,g in [(reader,'SENTENCE'),(reader,'PASSAGE'),('Qwen','SENTENCE'),('Qwen','PASSAGE')]]
                    result=difference_in_differences(cells,n=10000,seed=42)
                    interactions.append(dict(dataset=dataset,reader=reader,reference='Qwen',budget=budget,contrast='(sentence-passage)_reader - (sentence-passage)_Qwen',family='additional_posthoc_exploratory',**result))
    matched=pd.DataFrame(comparisons);units=pd.DataFrame(unit_comparisons);did=pd.DataFrame(interactions)
    matched.to_csv(root/'named_reader_comparisons.csv',index=False)
    units.to_csv(root/'named_granularity_comparisons.csv',index=False)
    did.to_csv(root/'named_reader_granularity_interactions.csv',index=False)
    peaks=[]
    for (dataset,reader),g in summary.groupby(['dataset','reader']):
        base=g[g.evidence_granularity=='NONE'].iloc[0]
        best=g[g.evidence_granularity!='NONE'].sort_values(['task_score','prompt_tokens','action_id'],ascending=[False,True,True]).iloc[0]
        peaks.append(dict(dataset=dataset,reader=reader,baseline=base.task_score,best=best.task_score,delta=best.task_score-base.task_score,
            granularity=best.evidence_granularity,selector=best.selector,budget=best.budget,HELP=best.HELP,HURT=best.HURT,evidence_tokens=best.actual_evidence_tokens,
            interpretation='format-confounded strict accuracy' if dataset=='SciFact' and reader!='Qwen' else 'retrospective pilot peak'))
    peaks=pd.DataFrame(peaks);peaks.to_csv(root/'reader_peak_summary.csv',index=False)
    frame['flag_empty']=(frame.evidence_granularity!='NONE')&frame.selected_unit_ids.map(len).eq(0)
    frame['flag_cap']=frame.generated_tokens.ge(128)
    frame['flag_invalid']=(frame.dataset=='SciFact')&(~frame.prediction.str.strip().str.upper().isin(['SUPPORT','REFUTE','NOT_ENOUGH_INFO']))
    flags=frame.groupby(['dataset','reader']).agg(empty_evidence_rows=('flag_empty','sum'),generation_cap_rows=('flag_cap','sum'),invalid_label_rows=('flag_invalid','sum')).reset_index()
    flags.to_csv(root/'flag_counts_by_reader_dataset.csv',index=False)
    examples=[]
    for dataset,reader,flag in [('SciFact','Mistral','flag_invalid'),('SciFact','Llama','flag_invalid'),('QASPER','Mistral','flag_cap'),('QASPER','Llama','flag_empty')]:
        r=frame[(frame.dataset==dataset)&(frame.reader==reader)&frame[flag]].sort_values(['query_id','action_id']).iloc[0]
        examples.append(dict(dataset=dataset,reader=reader,flag=flag,query_id=r.query_id,action_id=r.action_id,
            prediction_excerpt=r.prediction[:220].replace('\n',' '),score=r.task_score,evidence_tokens=r.actual_evidence_tokens,generated_tokens=r.generated_tokens))
    pd.DataFrame(examples).to_csv(root/'flagged_examples.csv',index=False)
    diagnostic=pd.read_csv(root/'label_compliance_diagnostic_summary.csv')
    diagnostic=diagnostic[(diagnostic.evidence_granularity=='NONE')|((diagnostic.evidence_granularity=='PASSAGE')&(diagnostic.selector=='semantic')&(diagnostic.budget.astype(str)=='1024'))|((diagnostic.evidence_granularity=='SENTENCE')&(diagnostic.selector=='semantic')&(diagnostic.budget.astype(str)=='256'))]
    oracle=pd.read_csv(root/'oracle_summary_df.csv')
    common=summary[(summary.selector=='semantic')&summary.evidence_granularity.isin(['PASSAGE','SENTENCE'])&summary.budget.astype(str).isin(['1024','FULL'])]
    common=common[['dataset','reader','evidence_granularity','budget','task_score','HELP','HURT','actual_evidence_tokens']].sort_values(['dataset','evidence_granularity','budget','reader'],key=lambda x:x.astype(str))
    text=['# Week 11 D — reader × evidence granularity: full research review',
        '**D is complete for the available text surface.** The mechanical audit passed all 16,650 checksummed rows. The experiment contains 150 unique development queries (50 per dataset), not 16,650 independent examples. QASPER has 40 paper clusters; HotpotQA and SciFact each have 50 query/claim clusters. Each query has 37 executable conditions per reader. The 1,800 configured token-compression rows remain pending LLMLingua-2 qualification.',
        '## What was tested',
        'Mistral, Qwen and Llama used the same frozen SFR/P0 top-20 candidate retrieval, passage/sentence/exact-clause proposition exposure, lexical/semantic/hybrid selectors, and budgets 256/512/1024/FULL. Every reader used exactly the reviewed C source selections, restored to source order. Numeric budgets enforce the maximum native-token count across all three readers. Realized token use can differ by tokenizer and by selected granularity. FULL is the frozen capped candidate context, not the entire paper or unlimited context.',
        'Each public reader retained its pinned model/revision, FP16 precision, native chat template, greedy 128-token decoding limit and frozen answer instructions. Qwen thinking remained disabled. Gold answers/evidence were used only for evaluation. Same-reader no-evidence controls define HELP/HURT (QA epsilon 0.01; classification exact).',
        '## Main findings',
        '1. **There is no universal best reader/granularity combination in this pilot.** Qwen has the largest HotpotQA point-estimate peak; Llama has the largest QASPER peak with FULL context. These peaks are selected on the same 50 queries and do not establish independently confirmed superiority.',
        '2. **Point estimates vary with reader, representation and budget, but a formal interaction is not established.** At semantic passage/1,024 tokens, QASPER Qwen and Llama are almost tied; Llama has a larger FULL-context point estimate. HotpotQA Qwen has a higher 1,024-token passage point than its own FULL point. Same-budget sentence exposure does not consistently improve over passages. All eight additional QA reader-by-granularity interaction intervals include zero.',
        '3. **SciFact is heavily confounded by output-format compliance.** Mistral has 1,849/1,850 invalid strict-format outputs; Llama has 1,826/1,850; Qwen has zero. Strict scores remain valid measurements of the frozen response contract, but they cannot support a clean conclusion about relative claim reasoning or evidence utilization. Explanations following labels are scored invalid by the preregistered exact-label rule.',
        '4. **The useful continuation is controlled characterization, not selecting a universal winner or training a controller.** Keep Qwen/SFR/P0 as the pre-existing primary control for F/G; run the prescribed E reader/compressor panel, and retain D as exploratory evidence of reader/context dependence.',
        '## Baselines and best observed condition per reader',
        'QA entries are answer F1; SciFact entries are strict classification accuracy. Each row has 50 queries. Best conditions are retrospective; different rows can use different evidence settings. These peaks are not matched-condition reader effects.',
        peaks.to_markdown(index=False,floatfmt='.6f'),
        '## Matched-condition results',
        'This table holds retriever, selector, granularity and requested budget fixed across readers. Identical source text does not imply identical native token counts. The reader-specific answer contract/template is part of the frozen reader interface.',
        common.to_markdown(index=False,floatfmt='.6f'),
        '## Paired reader comparisons',
        'The following additional analyses were specified after D completed and are labelled post-hoc exploratory. Every comparison pairs the same 50 queries; 10,000 resamples, seed 42; paper-cluster bootstrap for QASPER and query/claim bootstrap otherwise. Intervals are nominal, without multiple-comparison or winner-selection correction. A CI touching zero is not a clean positive result. SciFact intervals are about strict formatting-and-correctness performance, not isolated reasoning ability.',
        matched.to_markdown(index=False,floatfmt='.6f'),
        'For example, Llama minus Qwen on QASPER FULL passages is +0.034503 with CI [-0.031313, 0.098838]; on FULL sentences it is +0.053496 with CI [-0.015459, 0.124248]. Neither establishes superiority. On HotpotQA 1,024-token passages, Qwen minus Llama is +0.055476 with CI [-0.031552, 0.144145]; on 1,024-token sentences it is +0.136762 with nominal CI [0.037176, 0.241214]. The latter is an exploratory matched-condition signal, not a universal reader result.',
        '## Does sentence exposure beat passages at the same requested budget?',
        'The contrast is sentence minus passage with semantic selection. Actual token use is shown because requested-budget matching is not exact length matching. A positive point estimate alone is insufficient for superiority. FULL sentence/passage formatting and coverage can differ slightly even when nearly all source material is exposed.',
        units.to_markdown(index=False,floatfmt='.6f'),
        'Llama HotpotQA sentence minus passage at 1,024 requested tokens is -0.089465, nominal CI [-0.169546, -0.020935], under this allocation policy. Qwen and Mistral intervals for the same contrast include zero. A negative within-reader interval versus an inconclusive interval in another reader does not itself prove an interaction; the direct difference-in-differences test is required.',
        '## Reader × granularity interaction',
        'These eight additional post-hoc QA contrasts compare the sentence-minus-passage effect for a reader against the same effect for Qwen. They are separate from the twelve explicitly configured H interaction contrasts. Interpret their nominal CIs as exploratory; do not treat descriptive factorial marginal averages as causal effects.',
        did.to_markdown(index=False,floatfmt='.6f'),
        'All eight direct interaction CIs include zero. The observed patterns motivate further controlled study but do not justify a confirmed reader × granularity interaction claim on this cohort.',
        '## Flags and suspicious records',
        flags.to_markdown(index=False),
        'Counts above are logical condition rows, not independent queries. There are 924 empty-evidence rows: 308 shared query/condition selections repeated across three readers. This follows the frozen strict-prefix allocation rule: an oversized first unit stops selection, so coarse passages can expose nothing at small budgets. Preserve this limitation instead of changing the allocation after seeing outcomes.',
        'There are 987 generation-cap rows: Mistral 326, Llama 661, Qwen 0. Llama accounts for 649 SciFact cap hits; Mistral accounts for 264. This supports a response-format/verbosity concern. Cap hits can truncate otherwise meaningful answers; they were not silently retried with a larger limit. QA token F1 can also penalize explanatory verbosity, so reader F1 differences are not pure knowledge measurements.',
        'Seven QASPER queries contain incompletely mapped evidence annotations (four fully unmapped, three partially mapped). Their 777 flagged logical rows are 7 queries × 37 conditions × 3 readers. Annotation gaps constrain evidence-recall interpretation; answer-quality scoring is retained. SciFact has 25 no-evidence-label claims with null retrieval recall rather than fabricated zero recall; classification still uses all 50 claims.',
        'Illustrative saved records follow. Full query/action flags remain in `flagged_queries.csv`; excerpts do not replace the original predictions.',
        pd.DataFrame(examples).to_markdown(index=False),
        '## Separate post-hoc label-compliance diagnostic',
        'This diagnostic accepts only an unambiguous exact leading label followed by an allowed boundary/punctuation. It does not search explanations, paraphrase labels or modify primary scoring. It was added after inspecting format failures. The values below help identify a formatting confound; they are not a replacement accuracy leaderboard, confirmatory claim, or inference-time feature.',
        diagnostic[['reader','evidence_granularity','selector','budget','strict_accuracy','strict_format_rate','leading_label_rate','leading_label_diagnostic_accuracy','cap_hit_rate']].to_markdown(index=False,floatfmt='.6f'),
        '## Retrospective oracle headroom',
        oracle.to_markdown(index=False,floatfmt='.6f'),
        'ALL_READERS lets an oracle choose both reader and evidence condition using the observed answers. That is an optimistic retrospective ceiling and ignores the serving burden of swapping reader models. It does not show that a dynamic controller works. The strict SciFact oracle inherits the label-compliance confound. No controller was trained.',
        '## Cost and reproducibility',
        'D recorded 7,012 physical generation calls: 3,506 Mistral and 3,506 Llama. All 5,550 Qwen results exactly reuse C predictions and scores. Other identical-evidence cases also reuse cached answers. Logical LM calls remain per-policy costs; cached answers are not independent repetitions. C computation and operational qualification calls are separate from D physical-call counters.',
        'Reader process isolation changed operational placement only. Pinned weights, canonical generation, prompts, evidence and evaluation remain unchanged; supplemental driver/qualification receipts record the recovery. Reader-native prompt and generation tokens are retained. Cached original timing fields omit some preprocessing/IPC costs; measured end-to-end latency remains null. No serving-latency conclusion is justified from those fields.',
        '## Continuation decisions',
        '- D stays frozen with LLMLingua conditions explicitly pending. No existing prediction, score, source selection or status artifact was overwritten for this review.',
        '- E compares C0 raw top-10-pool passages and C2 exact query-ranked top-eight sentences; C1 remains pending. The QASPER/Mistral SARA specialist C3 has its own adapted no-evidence baseline and is reported separately. Because E uses width 10 while D uses width 20, controlled compressor claims must come from within E.',
        '- F uses the five manually reviewed actual C actions selected before D/E outcomes; the same settings are tested on every dataset with Qwen and BM25/MiniLM/SFR. Their selection remains exploratory. D does not trigger a silent reader/retriever change.',
        '- G keeps its prescribed Qwen/SFR hierarchy controls. H/I are saved-result analyses only. Prospective prompt/parser changes, a larger confirmation cohort, or dependency-qualified token compression require a new output version.',
        '## Complete 333-condition summary',
        'The complete fixed reader × selector × granularity × budget table follows. Metric scales remain dataset-specific; there is no pooled QA/verification overall score.',
        summary[['dataset','reader','evidence_granularity','selector','budget','task_score','delta_task_score','HELP','HURT','actual_evidence_tokens','prompt_tokens','generated_tokens','invalid_label_rate']].sort_values(['dataset','reader','evidence_granularity','selector','budget'],key=lambda x:x.astype(str)).to_markdown(index=False,floatfmt='.6f')]
    (root/'RESEARCH_REVIEW.md').write_text('\n\n'.join(text)+'\n')
    atomic(root/'research_review_provenance.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),source_status_sha256=file_hash(root/'status.json'),source_summary_sha256=file_hash(root/'reader_granularity_summary_df.csv'),driver_sha256=file_hash(Path(__file__)),additional_comparisons='post-hoc exploratory; frozen primary scores unchanged',comparisons=len(matched),granularity_comparisons=len(units),interactions=len(did),resamples=10000,seed=42))
    print(peaks.to_string(index=False))
    print('\nNamed Llama-minus-Qwen comparisons:')
    print(matched[(matched.reader_left=='Llama')&(matched.reader_right=='Qwen')&(matched.dataset!='SciFact')].to_string(index=False))


if __name__=='__main__':main()
