"""Read-only, dataset-specific research interpretation of the completed E panel."""
from pathlib import Path
import sys,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.week11.protocol import load,gate,output
from src.week11.cache import Store,file_hash,atomic
from src.week11.bootstrap import paired


def main():
    cfg=load();gate(cfg,'F');root=output(cfg,'E');manifest=json.loads((root/'manifest.json').read_text())
    rows=Store(root/'rows',manifest['identity']).rows();assert len(rows)==4150
    frame=pd.DataFrame(rows);summary=pd.read_csv(root/'compressor_reader_summary_df.csv');assert len(summary)==83
    def group(dataset,reader,compressor,budget):
        g=frame[(frame.dataset==dataset)&(frame.reader==reader)&(frame.compressor==compressor)&(frame.budget.astype(str)==str(budget))]
        assert len(g)==50
        return g.to_dict('records')
    comparisons=[];cross=[];saturation=[]
    for dataset in cfg['datasets']:
        for reader in cfg['readers']:
            for budget in cfg['budgets']:
                a=group(dataset,reader,'C2',budget);b=group(dataset,reader,'C0',budget)
                byid={r['query_id']:r for r in b}
                assert all(r['candidate_pool_ids']==byid[r['query_id']]['candidate_pool_ids'] and r['source_hash']==byid[r['query_id']]['source_hash'] for r in a)
                result=paired(a,b,n=10000,seed=42)
                comparisons.append(dict(dataset=dataset,reader=reader,budget=budget,contrast='C2_minus_C0',C2_score=sum(r['task_score'] for r in a)/50,C0_score=sum(r['task_score'] for r in b)/50,
                    C2_evidence_tokens=sum(r['actual_evidence_tokens'] for r in a)/50,C0_evidence_tokens=sum(r['actual_evidence_tokens'] for r in b)/50,**result))
            a=group(dataset,reader,'C2',256)
            for budget in [512,1024,'FULL']:
                b=group(dataset,reader,'C0',budget);result=paired(a,b,n=10000,seed=42)
                cp=sum(r['prompt_tokens'] for r in a)/50;rp=sum(r['prompt_tokens'] for r in b)/50
                cross.append(dict(dataset=dataset,reader=reader,C2_budget=256,C0_budget=budget,C2_score=sum(r['task_score'] for r in a)/50,C0_score=sum(r['task_score'] for r in b)/50,
                    C2_prompt_tokens=cp,C0_prompt_tokens=rp,prompt_reduction=1-cp/rp,analysis='posthoc_cross_budget_descriptive',**result))
            a=group(dataset,reader,'C2',1024);b={r['query_id']:r for r in group(dataset,reader,'C2','FULL')}
            saturation.append(dict(dataset=dataset,reader=reader,exact_evidence_matches=sum(r['evidence_hash']==b[r['query_id']]['evidence_hash'] for r in a),identical_predictions=sum(r['prediction']==b[r['query_id']]['prediction'] for r in a),queries=50))
    named=pd.DataFrame(comparisons);cost=pd.DataFrame(cross);sat=pd.DataFrame(saturation)
    named.to_csv(root/'named_compressor_comparisons.csv',index=False);cost.to_csv(root/'cross_budget_diagnostic.csv',index=False);sat.to_csv(root/'sentence_cap_saturation.csv',index=False)
    specialist=paired(group('QASPER','Mistral','C3','FULL'),group('QASPER','Mistral','C3',0),n=10000,seed=42)
    atomic(root/'specialist_vs_own_baseline.json',specialist)
    frame['empty']=(frame.evidence_granularity!='NONE')&frame.selected_unit_ids.map(len).eq(0)
    frame['cap']=frame.generated_tokens.ge(128)
    frame['invalid']=(frame.dataset=='SciFact')&(~frame.prediction.str.strip().str.upper().isin(['SUPPORT','REFUTE','NOT_ENOUGH_INFO']))
    flags=frame.groupby(['dataset','reader']).agg(empty_evidence_rows=('empty','sum'),generation_cap_rows=('cap','sum'),invalid_label_rows=('invalid','sum')).reset_index()
    flags.to_csv(root/'flag_counts_by_reader_dataset.csv',index=False)
    peaks=[]
    for (dataset,reader,compressor),g in summary[(summary.evidence_granularity!='NONE')&(summary.compressor!='C3')].groupby(['dataset','reader','compressor']):
        r=g.sort_values(['task_score','prompt_tokens','action_id'],ascending=[False,True,True]).iloc[0]
        peaks.append(dict(dataset=dataset,reader=reader,compressor=compressor,budget=r.budget,task_score=r.task_score,HELP=r.HELP,HURT=r.HURT,evidence_tokens=r.actual_evidence_tokens,prompt_tokens=r.prompt_tokens))
    peaks=pd.DataFrame(peaks);peaks.to_csv(root/'public_compressor_peak_summary.csv',index=False)
    cols=['dataset','reader','reader_variant','compressor','evidence_granularity','budget','task_score','delta_task_score','HELP','HURT','actual_evidence_tokens','prompt_tokens','generated_tokens']
    text=['# Week 11 E — compressor × reader: full research review',
        '**E is complete for the available surface:** 4,150 checksummed logical rows, 83 dataset/condition summaries, and the same 150 unique development queries as C/D. The primary scores and frozen status are unchanged. Optional C1/LLMLingua-2 conditions (1,800 configured rows) remain pending. C4/faithful SeleCom is future work, not an executed experiment.',
        '## Experiments and controls',
        '| Policy | Executed mechanism | Scope |\n|---|---|---|\n| C0 | Raw passage evidence, semantic ranking, strict budget prefix | All three readers/datasets; 256/512/1024/FULL |\n| C1 | LLMLingua-2 token compression | Pending qualification; no results |\n| C2 | Query-only SFR ranking of exact source sentences, retain at most eight, restore source order, enforce budget | All three readers/datasets; 256/512/1024/FULL |\n| C3 | Existing adapted SARA Mistral checkpoint: top-four text passages plus remaining candidate vectors | QASPER/Mistral only, natural FULL point and own adapted no-evidence baseline |',
        'C0/C2 share the exact SFR/P0 candidate pool (retrieval width 10, capped by the frozen shared-context policy), query and source hash. This was checked for all 1,800 paired C2/C0 query conditions. C2 is non-generative extraction and mechanically source-grounded. Gold labels never select source units. A numeric budget is a shared tokenizer maximum, not an exact realized-length match. The rank-prefix rule can leave C0 empty when its first passage is too large.',
        'E uses pool width 10; D used width 20. Therefore D-to-E raw score differences cannot be attributed only to compression. Controlled compressor comparisons belong within E. C3 also changes checkpoint/interface/prompt construction, so its result is separate from the portable public-reader comparison.',
        '## Main findings',
        '1. **Sentence filtering is useful under tight budgets, but it is not a universal quality improvement.** C2 avoids many low-budget empty/coarse contexts and can expose a compact selection. With more budget, raw passages often retain higher QA point estimates. The contrast includes budget fit and unit exposure; it does not isolate a causal effect of removing irrelevant text alone.',
        '2. **The top-eight cap saturates.** C2 FULL is still at most eight sentences, not the full candidate context. Identical outputs at different requested budgets can be expected cache reuse, not a stalled experiment. The exact identity table below checks this directly.',
        '3. **The QASPER SARA specialist is a strong separate reference.** Its F1 is 0.403749 versus its own adapted no-evidence F1 0.190914. Its gain is not evidence that generic token compression or latent vectors alone caused the improvement: checkpoint adaptation, prompt/interface and exposed information differ from C0/C2.',
        '4. **The SciFact formatting confound persists.** Mistral and Llama strict classification results are dominated by violations of the exact-label contract. Preserve the primary scores and examine the separately labelled compliance diagnostic; do not infer that zero strict accuracy means no claim reasoning ability.',
        '## Best observed public-reader condition within each compressor',
        'Each row is a retrospective peak chosen on the same 50-query pilot. QA uses F1; SciFact uses strict accuracy. These are descriptive peaks, not independently confirmed selections.',
        peaks.to_markdown(index=False,floatfmt='.6f'),
        '## Matched-budget C2 minus C0 comparisons',
        'All rows pair the same queries under the same reader and candidate pool. Paired bootstrap: 10,000 resamples, seed 42; QASPER paper clusters (40), other datasets query/claim clusters (50). Intervals are nominal and exploratory. Actual evidence tokens are shown because equal requested budgets do not guarantee equal lengths.',
        named.to_markdown(index=False,floatfmt='.6f'),
        'The strongest primary-reader low-budget signals are QASPER C2/256 minus C0/256: +0.109351 F1, nominal CI [0.046333, 0.171307], and SciFact: +0.32 accuracy, CI [0.14, 0.50]. In contrast, HotpotQA +0.100422 F1 has CI [-0.005202, 0.210173] and is inconclusive. QASPER/SciFact C0 at that budget expose only about 19 mean evidence tokens, versus roughly 187–199 for C2, so budget-fit failures contribute to the contrast.',
        '## Cross-budget cost diagnostic',
        'These additional post-hoc contrasts compare C2 at 256 requested tokens against larger C0 budgets. Smaller prompt counts are a measured token-burden difference, not a serving-latency claim. Similar point estimates with broad CIs do not prove equivalence or non-inferiority. A CI containing zero does not establish no quality loss.',
        cost.to_markdown(index=False,floatfmt='.6f'),
        'Relative to Qwen C0/1024, C2/256 reduces mean final prompt tokens by 70.5% on QASPER, 70.9% on HotpotQA and 66.2% on SciFact. The corresponding quality-difference intervals allow losses of about 0.047 F1, 0.126 F1 and 0.10 accuracy respectively. These token savings therefore do not establish quality-preserving compression.',
        'For Qwen, C2/256 reaches HotpotQA F1 0.437931 versus C0/1024 0.480272, QASPER 0.206276 versus 0.199183, and SciFact strict accuracy 0.70 versus 0.68. These mixed point-estimate tradeoffs support a compact-context hypothesis, not a universal compression win. The paired intervals above determine how much uncertainty remains.',
        '## Top-eight saturation check: C2/1024 versus C2/FULL',sat.to_markdown(index=False),
        'A matching evidence hash means exact equality of the generator evidence text. Prediction equality is reported separately. This comparison never treats cached repeated outputs as independent replications.',
        '## QASPER-only SARA specialist',
        summary[summary.compressor=='C3'][cols].to_markdown(index=False,floatfmt='.6f'),
        pd.DataFrame([specialist]).to_markdown(index=False,floatfmt='.6f'),
        'This interval compares C3 evidence intervention with the same adapted checkpoint no-evidence baseline. It is a natural policy contrast, not a matched public-reader compression ablation. SARA uses a distinct canonical prompt/latent interface, trained projector and restored added-token weights. Its text evidence tokens exclude non-text vector information; final prompt tokens include the actual interface tokens. Source-span support is not asserted for projected latent information. A future matched adapted full-text/latent ablation is needed to isolate the contribution of latents.',
        '## Flags',flags.to_markdown(index=False),
        'There are 297 empty-evidence rows, 214 generation-cap rows, 900 invalid strict-label rows and 203 annotation-mapping rows. These are logical condition counts; the 203 mapping flags correspond to the same seven QASPER queries across 27 public conditions and two specialist conditions. Empty evidence follows the frozen allocation policy. Cap hits and invalid labels were preserved, not silently reparsed or regenerated.',
        'Full row/action flags are in `flagged_queries.csv`. The separately saved `label_compliance_diagnostic_summary.csv` is post-hoc and does not replace the primary scoring, bootstrap or oracle tables.',
        '## Cost and timing',
        '850 physical generation calls served 4,150 logical rows: Mistral 350 (including the specialist), Qwen 250 and Llama 250. The other 3,300 rows reused compatible cached outputs. This does not make their conceptual inference cost zero. C2 needs no extraction LM; SFR embedding preprocessing/ranking still has a conceptual cost. Candidate text in memory is not counted as LM prompt input. Operational qualification calls are recorded separately from scientific rows.',
        'Cached original retrieval/selection/generation timings do not measure full serving latency; model loading, IPC and some preparation are outside them. End-to-end latency remains unmeasured/null. No wall-clock efficiency superiority is claimed from these fields.',
        '## Continuation',
        'E remains frozen without changing any primary policy. F is the previously reviewed Qwen finalist panel under BM25/MiniLM/SFR; no post-hoc SARA promotion or reader swap is added. G retains Qwen/SFR to isolate deterministic hierarchy exposure. H/I use saved results only. The current 50-per-dataset development pilot needs a new, larger confirmation protocol before generalization claims.',
        '## Complete 83-condition result table',summary[cols].sort_values(['dataset','reader','compressor','budget'],key=lambda x:x.astype(str)).to_markdown(index=False,floatfmt='.6f')]
    (root/'RESEARCH_REVIEW.md').write_text('\n\n'.join(text)+'\n')
    atomic(root/'research_review_provenance.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),source_status_sha256=file_hash(root/'status.json'),driver_sha256=file_hash(Path(__file__)),paired_compressor_comparisons=36,additional_cross_budget_comparisons=27,primary_scores_changed=False))
    print('Named Qwen comparisons:');print(named[named.reader=='Qwen'].to_string(index=False));print('Specialist:',specialist)


if __name__=='__main__':main()
