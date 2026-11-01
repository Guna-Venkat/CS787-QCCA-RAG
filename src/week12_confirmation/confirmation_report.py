"""Separate task results; uncertainty-led interpretation, never selection after outcomes."""
from .freeze_guard import FREEZE_SHA,SPLIT_SHA

def claim_table(contrasts):
    def status(right,budget=None):
        xs=[r for r in contrasts.to_dict('records') if r['right']==right and (budget is None or r['budget']==budget)]
        positive=sum(r['ci_low']>0 for r in xs);negative=sum(r['ci_high']<0 for r in xs)
        return 'Supported by confirmation' if positive==len(xs) and positive else 'Partially supported / task-dependent' if positive else 'Not supported / inconclusive'+('; opposing effects present' if negative else '')
    return [
      ['Tight-budget sentence evidence helps','Strong simple constrained-context VAL policy',status('NONE',256)],
      ['Granularity depends on dataset and budget','VAL directions differ across budget/task','Descriptive pattern; interpret cell magnitudes and intervals below; not a newly tested interaction'],
      ['Large-budget passages can recover','RAW_SKIP competitive/stronger at 1024','Partially supported' if any(r['right']=='RAW_SKIP' and r['budget']==1024 and r['point']<=0 for r in contrasts.to_dict('records')) else 'Not supported / inconclusive'],
      ['QBGF universally superior','Not established on VAL',('Confirmation favors QBGF across all four cells; challenges broad generalization of the VAL rejection, without retrospective promotion' if all(r['ci_high']<0 for r in contrasts.to_dict('records') if r['right']=='QBGF_FULL') else 'Mixed/tied or opposing cell evidence; no robust universal promotion established. Read each secondary interval below')],
      ['Query conditioning matters','D VAL mechanism evidence','Validation-supported only; not tested in confirmation']]

def write(out,frames,synthetic=False):
    title='SYNTHETIC PIPELINE TEST — NOT CONFIRMATION RESULTS' if synthetic else 'Week12 M — final untouched confirmation'
    lines=['# '+title,'','## Integrity and untouched boundary','',f'Freeze SHA256: `{FREEZE_SHA}`. Split SHA256: `{SPLIT_SHA}`.','', 'Frozen final method remains FIXED_SENTENCE; SFR top-10 → Qwen/Qwen3-8B BASE_FP16; no adaptation. Five systems at 256/1024. No selection changes are permitted after this evaluation.','', 'Real cohort: HotpotQA 200 queries / 200 groups; SciFact 50 claims / 43 groups; QASPER zero fresh confirmation. Historical QASPER TEST remains closed.','', 'This implementation and its synthetic tests observed no confirmation outcomes before explicit full execution. The full-execution receipt records the first authorized access by this workflow. This statement does not certify unavailable external execution history.','', '## Results','']
    for d in ['HotpotQA','SciFact']:
        lines+=['### '+d,'',frames['summary'].query('dataset == @d').to_markdown(index=False,floatfmt='.5f'),'']
    lines+=['### SciFact class diagnostics','',frames['scifact_classes'].to_markdown(index=False,floatfmt='.5f'),'','## Paired predefined confirmation contrasts','', 'Nominal 95% paired frozen-group bootstrap intervals; 10,000 resamples, seed 42. Primary family: 8 contrasts; secondary family: 4 contrasts; separate Holm-adjusted paired group label-swap p-values. H3 is descriptive. Synthetic tests use only 100 resamples. Interpret direction, magnitude, uncertainty and consistency; no project-level PASS/FAIL.','',frames['contrasts'].to_markdown(index=False,floatfmt='.5f'),'','## Validation versus confirmation','',frames['validation_comparison'].to_markdown(index=False,floatfmt='.5f'),'','Changes between VAL and confirmation are descriptive and unpaired: samples differ. No pooled cross-dataset quality score.','', '## Scientific conclusions','', '| Claim | Validation evidence | Confirmation evidence / final status |','|---|---|---|']
    if synthetic:lines+=['| All scientific claims | Synthetic fixtures only | Pending real confirmation; no inference |']
    else:
        for r in claim_table(frames['contrasts']):lines+=['| '+' | '.join(r)+' |']
        for r in frames['contrasts'].to_dict('records'):
            direction='positive' if r['point']>0 else 'negative' if r['point']<0 else 'zero'
            lines += [f"- {r['dataset']} @{r['budget']}: FIXED_SENTENCE − {r['right']} = {r['point']:.5f} [{r['ci_low']:.5f}, {r['ci_high']:.5f}], {direction}. " + ('Interval includes zero; inconclusive direction.' if r['ci_low']<=0<=r['ci_high'] else 'Direction supported by nominal interval; see multiplicity-adjusted evidence.')]
    lines+=['','Costs are conceptual query-time tokens/calls; exact-answer reuse does not turn a system into zero-cost inference. Generation timing is recorded for the cached answer and is not a new timing measurement per reused row. No end-to-end latency, serving throughput or wall-clock efficiency claim is made.','', '## Limitations','', '- Only 250 confirmation queries, from HotpotQA and SciFact.', '- SciFact has only 50 claims / 43 frozen groups; intervals may be wide.', '- No fresh QASPER confirmation; consumed historical TEST remains closed.', '- D query-conditioning mechanism remains validation-only.', '- J/K/L attribution is a separate post-hoc VAL diagnostic and did not change M.', '- LLMLingua and LoRA were not evaluated in confirmation.', '- Statistical intervals and corrected tests complement effect size and direction; absence of significance is not equivalence.','']
    (out/'week12_M_final_confirmation_report.md').write_text('\n'.join(lines))
