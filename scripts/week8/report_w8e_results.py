"""Summarize the frozen controller; paired intervals do not change its selection."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from src.week8.selective_controller import canonical_baselines
from src.week8.closeout import cached_split


def paired_interval(frame,differences,n=10000):
    t=pd.DataFrame({'paper':frame.paper_id.to_numpy(),'d':np.asarray(differences)})
    g=t.groupby('paper').d.agg(['sum','count'])
    ix=np.random.default_rng(42).integers(0,len(g),size=(n,len(g)))
    values=g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1)
    return {'difference':float(t.d.mean()),'ci95':np.quantile(values,[.025,.975]).tolist()}


def main():
    out=ROOT/'results/week8/w8e_controller'
    summary=json.loads((out/'controller_summary.json').read_text())
    selected=summary['selected_policy'];replays={}
    for name in summary['internal_val_metrics']:
        replays[name]=pd.read_csv(out/f'{name}_internal_val_replay.csv',dtype={'question_id':str,'paper_id':str}).sort_values('question_id').reset_index(drop=True)
    baseline=replays['E1'];comparisons={}
    for name,r in replays.items():
        if r.question_id.tolist()!=baseline.question_id.tolist():raise ValueError('Unpaired controller queries')
        comparisons[name]={'f1_vs_E1':paired_interval(r,r.final_f1.to_numpy()-baseline.final_f1.to_numpy()),
            'hurt_vs_E1':paired_interval(r,(r.realized_delta_f1.to_numpy()<-.01).astype(float)-(baseline.realized_delta_f1.to_numpy()<-.01).astype(float))}
    chosen=replays[selected];static=canonical_baselines(cached_split('internal_val'))
    selected_static={}
    for k in [4,6]:
        reference=static[static.k==k].set_index('question_id').f1
        selected_static[f'k{k}']=paired_interval(chosen,chosen.final_f1.to_numpy()-chosen.question_id.map(reference).to_numpy())
    (out/'paired_paper_intervals.json').write_text(json.dumps({'resamples':10000,'fixed_prediction_paper_bootstrap':True,
        'family_comparisons_vs_always_DraftSemantic':comparisons,'FIT_selected_policy':selected,
        'selected_policy_vs_static':selected_static,'no_reselection_from_intervals':True},indent=2))
    fit=summary['fit_metrics'];val=summary['internal_val_metrics']
    text='''# W8-E frozen controller results — 18 October 2026

Controller construction used measured canonical-draft U1–U12 token diagnostics, gain-weighted logistic regression (B0-GW), and five paper-grouped FIT OOF folds. Gain weights were fixed at 1+5 max(0,best_delta). The original W8-B validation-only/random-fold gate results were not reused as FIT predictions.

The score and safety thresholds are chosen on FIT OOF realized downstream F1, not AUC or an oracle-assisted objective. Within 0.003 of the highest F1, lower cumulative prompt tokens are preferred, then cached-generation latency, then simplicity. Family configurations and the overall selected policy were saved before the controller validation confirmation. Previous INTERNAL-VAL inspection and the W8-D closeout are acknowledged; this is development confirmation, not an untouched final test.

'''
    text+=f"**FIT-selected policy: {selected}.** The validation table does not change that selection.\n\n"
    text+=f"The full-grid best FIT F1 is {summary['fit_grid_best_downstream_f1']:.6f}; the selected point is {summary['fit_selected_shortfall_from_grid_best']:.6f} below it. The 0.003 tolerance is applied once globally. Other family rows show their own cost-efficient FIT operating points.\n\n"
    text+='| Policy | FIT OOF F1 | Validation F1 | Promotion rate | HURT / all queries | Mean cumulative prompt tokens |\n|---|---:|---:|---:|---:|---:|\n'
    for name,m in val.items():text+=f"| {name} | {fit[name]['downstream_f1']:.6f} | {m['downstream_f1']:.6f} | {m['promotion_rate']:.2%} | {m['hurt_rate_all_queries']:.2%} | {m['mean_cumulative_prompt_tokens']:.2f} |\n"
    text+='''
E0 returns k4; E1 always promotes DraftSemantic; E2 uses B0-GW + DraftSemantic; E3 uses B0-GW + frozen U4-A; E5 variants add predicted HELP/HURT vetoes. E4 was omitted according to the earlier W8-D closeout decision. There is no second-promotion experiment.

## Selected operating point

'''
    m=val[selected]
    text+=f"Configuration: `{json.dumps(summary['selected_config'],sort_keys=True)}`\n\n"
    text+=f"Validation F1 {m['downstream_f1']:.6f}; gain over k4 {m['gain_over_k4']:+.6f}; gain over k6 {m['gain_over_k6']:+.6f}. Promotion rate {m['promotion_rate']:.2%}. HELP among promotions {m['help_precision_promotions']:.2%}; HURT among promotions {m['hurt_rate_promotions']:.2%}. All-query HURT is {m['hurt_rate_all_queries']:.2%}.\n\n"
    text+=f"Exact mean final prompt tokens: {m['mean_final_prompt_tokens']:.2f}. Exact cumulative prompt tokens: {m['mean_cumulative_prompt_tokens']:.2f}. Both include the actual chat template and special tokens; promoted queries count two generation prompts.\n\n"
    text+='## Paired paper intervals for the FIT-selected policy\n\n10,000 paper resamples of fixed validation predictions; query-weighted differences. These intervals do not include variation from retraining or correct for all prior development experiments.\n\n'
    for label,result in [('always-promote DraftSemantic',comparisons[selected]['f1_vs_E1']),*selected_static.items()]:
        text+=f"- Versus {label}: F1 difference {result['difference']:+.6f}, 95% interval [{result['ci95'][0]:+.6f}, {result['ci95'][1]:+.6f}].\n"
    e1=val['E1'];k6=summary['internal_val_reference_baselines']['6']
    text+=f"\nThe frozen policy uses {1-m['mean_cumulative_prompt_tokens']/e1['mean_cumulative_prompt_tokens']:.2%} fewer cumulative prompt tokens than always-promote DraftSemantic and lowers all-query HURT by {e1['hurt_rate_all_queries']-m['hurt_rate_all_queries']:.2%}, but its F1 is lower. The selected policy's paired F1 interval versus always promoting is entirely below zero; intervals versus k4 and k6 include zero. There is no confirmed quality gain over either static baseline.\n\n"
    text+=f"Static k6 has F1 {k6['f1']:.6f} and {k6['tokens']:.2f} mean prompt tokens. The selected policy has {m['mean_cumulative_prompt_tokens']/k6['tokens']-1:+.2%} cumulative prompt cost versus k6 and lower point-estimate F1. The present result does not establish a better quality/cost operating point than k6. Smaller final prompts alone would miss the cost of the initial draft generation. These are confirmation results; they do not authorize switching the frozen policy to a validation winner.\n"
    text+='''
## What the result establishes

Decisions use only measured draft features and model predictions. Selected outcomes are looked up after the action; harmful outcomes are retained. A STOP action uses the base answer, not an oracle choice between outputs. Query ID, paper ID, original passage index, semantic rank, BM25 rank, proposed identity and selected action are preserved in the saved decision/replay records.

The replay uses real saved deterministic single-promotion generations. It does not rerun SARA for every threshold combination. Base and chosen promotion generation latency are summed from measured generation records. This is explicitly an estimated generation latency, **not measured end-to-end deployment latency**: routing, gate and selector overhead are not invented or omitted under an end-to-end label. A complete runtime/compute dominance claim still needs a deployed benchmark.

The original W8-C selector and results are unchanged. Invalid synthetic controller tuning outputs, including archived copies, were removed. Missing real inputs raise errors; no synthetic fallback is available. TEST remains untouched. Re-running the completed notebook verifies source hashes and inspects the frozen run rather than silently retuning on validation.
'''
    (out/'RESULTS.md').write_text(text)
    print(out/'RESULTS.md')
    print('FIT-selected policy:',selected)
    print(json.dumps(m,indent=2))

if __name__=='__main__':main()
