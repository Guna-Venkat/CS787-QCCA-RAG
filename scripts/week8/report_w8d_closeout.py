"""Render measured closeout results without hardcoded winners or point estimates."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd


def main():
    out=ROOT/'results/week8/w8d_closeout';r=json.loads((out/'closeout_summary.json').read_text())
    text='''# W8-D cached-feature closeout — 18 October 2026

The original extraction was repaired for question whitespace on 23 FIT and 5 INTERNAL-VAL queries. Canonical W7/W8-A generation strips questions; the initial masking code did not. Only those masking measurements were repeated. Original valid MiniLM semantic components and all W8-C model inputs were retained, and the closeout was rerun using the same fixed protocol. Superseded results and a repair ledger are preserved under `results/week8/archive_w8d_question_normalization/`.

No new attribution architecture search was performed. S5 ranks masking attribution (per token; equivalent to sum within one query), S6 ranks MIG, S7 adds alpha*attribution to DraftSemantic, and S8 also adds beta*MIG. Alpha=[0,.1,.3,1,3], beta=[0,.1,.3,1]. Each FIT outer fold tunes coefficients on training papers only. Final coefficients are frozen from all FIT before development confirmation.

U4-B uses the existing frozen encoder/fusion architecture plus masking sum, per-token masking, target count and availability. Duplicate A6 is excluded. Imputation and standardization are fitted within training papers; every optimizer step contains one query's six candidates. Seed42, five paper folds and three epochs are unchanged. A matched U4-A replay reproduced original OOF scores within 1.4e-7 before correction; unchanged BASE inputs/encoder features are verified before reusing this reference.

'''
    for name,p in r['parameters'].items():text+=f"Frozen {name} coefficients: `{json.dumps(p['frozen_full_fit_parameters'],sort_keys=True)}`.\n\n"
    for split,key in [('FIT OOF','fit_metrics'),('INTERNAL-VAL development confirmation','internal_val_metrics')]:
        text+=f'## {split}\n\n| Selector | Always-promote F1 | HELP | HURT | Mean regret | Oracle-gated SHR |\n|---|---:|---:|---:|---:|---:|\n'
        for name,m in r[key].items():text+=f"| {name} | {m['always_promote_f1']:.6f} | {m['help_rate']:.2%} | {m['hurt_rate']:.2%} | {m['mean_regret']:.6f} | {m['oracle_gated_shr']:.2%} |\n"
        text+='\n'
    text+='Oracle-gated SHR is explicitly a selector diagnostic. Always-promote F1 is the selection-only deployable quantity; no learned STOP performance is claimed here.\n\n'
    text+='## Escalation-needed validation subset\n\nTrue best candidate delta >0.01 identifies this retrospective diagnostic subset; it is never an online gate.\n\n| Selector | Queries | HELP | HURT | Mean delta | Mean regret | Best selected (ties accepted) |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name,m in r['internal_val_metrics'].items():
        e=m['escalation'];text+=f"| {name} | {e['n']} | {e['help_rate']:.2%} | {e['hurt_rate']:.2%} | {e['mean_delta']:.6f} | {e['mean_regret']:.6f} | {e['top1_tied_best']:.2%} |\n"
    text+='\n## Paired paper bootstrap versus DraftSemantic\n\n10,000 paper resamples, retaining every query in each sampled paper. Query-weighted estimates of fixed predictions; intervals do not include retraining variation or correct for all development experiments.\n\n| Selector | F1 difference | 95% F1 interval | HURT difference | 95% HURT interval |\n|---|---:|---:|---:|---:|\n'
    for name,m in r['internal_val_paper_bootstrap_vs_S3'].items():
        f=m['f1'];h=m['hurt'];text+=f"| {name} | {f['difference']:+.6f} | [{f['ci95'][0]:+.6f}, {f['ci95'][1]:+.6f}] | {h['difference']:+.2%} | [{h['ci95'][0]:+.2%}, {h['ci95'][1]:+.2%}] |\n"
    scores=pd.read_csv(out/'internal_val_selector_scores.csv',dtype={'question_id':str}).sort_values(['question_id','semantic_rank'])
    baseline=scores.loc[scores.groupby('question_id').S3.idxmax()].set_index('question_id').candidate_passage_index
    text+='\n'
    for name in ['S7','S8']:
        selection=scores.loc[scores.groupby('question_id')[name].idxmax()].set_index('question_id').candidate_passage_index
        text+=f"{name} changes {int((selection!=baseline).sum())} validation passage identities relative to DraftSemantic.\n\n"
    decision=r['decision'];candidate=decision['candidate_frozen_on_fit']
    text+=f"## Frozen decision\n\nThe confirmation candidate selected on FIT OOF was **{candidate}**. The corrected closeout verdict is **{decision['verdict']}**. Online attribution retained: **{decision['online_attribution_survives']}**.\n\n"
    text+='''The success rule was fixed before confirmation: F1 gain >=0.003 with a positive lower paired interval, or HURT reduction >=0.02 with a strictly negative upper interval and F1 noninferiority (lower interval >-0.003). Validation is not used to replace the FIT-selected candidate with whichever variant looks best afterward.

A lack of demonstrated gain in these tested variants is not proof that attribution can never help. Support for an existing answer while compressed and the counterfactual value of exposing a passage at full fidelity are distinct measurements. Negative masking means zeroing the SFR vector increased the likelihood of the existing draft, not that removing the passage improves correctness.

The original full extraction recorded about 8.8 hours of feature work. Targeted repair adds only the 28 affected queries. Any online attribution policy must account for seven SARA forward passes per nonempty draft; tiny unconfirmed gains do not establish operational value. Full results, candidate scores, fold coefficients, paired intervals and U4-B weights remain in this directory. Original W8-C checkpoints and outputs are unchanged. TEST remains untouched.
'''
    (out/'CLOSEOUT.md').write_text(text)
    print(out/'CLOSEOUT.md')
    print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
