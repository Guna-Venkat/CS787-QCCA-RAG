"""Audit a completed generation phase without changing scientific results."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.week11.protocol import load, gate, output, code_identity, scientific_config
from src.week11.cache import Store, digest, atomic, file_hash
from src.week11.datasets import source_identity
from src.week11.planning import plan, pending_reason, action


def audit(phase):
    cfg = load(); gate(cfg, phase); root = output(cfg, phase)
    status = json.loads((root/'status.json').read_text())
    assert status['executable_complete'] and not status['synthetic']
    manifest = json.loads((root/'manifest.json').read_text()); identity = manifest['identity']
    assert identity['code_hash'] == digest(code_identity())
    assert identity['config_hash'] == digest(scientific_config(cfg, phase))
    assert identity['dataset_hash'] == digest(source_identity(cfg))
    rows = Store(root/'rows', identity).rows()
    cohort = json.loads((root/'cohort.json').read_text())['selected']
    frozen_cohort = json.loads((output(cfg, 'C')/'cohort.json').read_text())['selected']
    assert cohort == frozen_cohort
    expected = set()
    for e in cohort:
        actions = [a for a in plan(cfg, phase, e['dataset']) if not pending_reason(a, cfg)]
        for reader in sorted({a['reader'] for a in actions}):
            baseline = action(cfg,reader=reader,granularity='NONE',selector='none',budget=0,pool_width=10 if phase=='E' else cfg['retrieval_width'])
            if not any(a['action_id']==baseline['action_id'] for a in actions):
                actions.append(baseline)
        expected.update((e['dataset'],e['query_id'],a['action_id']) for a in actions)
    actual = [(r['dataset'],r['query_id'],r['action_id']) for r in rows]
    assert set(actual) == expected and len(actual) == len(expected) == status['expected'] == status['completed']
    baselines = {(r['dataset'],r['query_id'],r['reader'],r['reader_variant']):r for r in rows if r['evidence_granularity']=='NONE'}
    flag_rows = []
    for r in rows:
        assert r['code_hash'] == identity['code_hash'] and r['config_hash'] == identity['config_hash']
        assert not r['synthetic'] and r['total_measured_latency'] is None
        assert 0 <= r['task_score'] <= 1
        assert r['budget']=='FULL' or r['actual_evidence_tokens'] <= r['budget']
        assert r['compressor']=='C3' or r['source_support']==1
        base = baselines[r['dataset'],r['query_id'],r['reader'],r['reader_variant']]
        epsilon = 0 if r['dataset']=='SciFact' else cfg['epsilon']
        delta = r['task_score']-base['task_score']
        assert abs(r['delta_task_score']-delta) < 1e-12
        assert r['HELP']==(delta>epsilon) and r['HURT']==(delta < -epsilon)
        assert r['answer_changed']==(r['prediction'].strip()!=base['prediction'].strip())
        flags = []
        if r['evidence_granularity']!='NONE' and not r['selected_unit_ids']:flags.append('empty_evidence_strict_prefix')
        if r['dataset']=='SciFact' and r['prediction'].strip().upper() not in cfg['datasets']['SciFact']['labels']:flags.append('invalid_verification_label')
        if r['generated_tokens'] >= cfg['generation']['max_new_tokens']:flags.append('generation_cap_reached')
        if r['unmapped_gold']:flags.append('unmapped_gold_annotation')
        if flags:flag_rows.append({k:r[k] for k in ['dataset','query_id','reader','action_id','evidence_granularity','selector','budget','task_score','HURT']} | {'flags':';'.join(flags)})
    reused = 0
    if phase=='D':
        croot = output(cfg,'C'); ci=json.loads((croot/'manifest.json').read_text())['identity']
        crows=Store(croot/'rows',ci).rows()
        match=lambda r:(r['dataset'],r['query_id'],r['evidence_granularity'],r['selector'],str(r['budget']))
        original={match(r):r for r in crows}
        for r in rows:
            c=original[match(r)]
            assert r['selection_hash']==c['selection_hash'] and r['evidence_hash']==c['evidence_hash']
            assert r['selected_unit_ids']==c['selected_unit_ids']
            if r['reader']=='Qwen':
                assert r['prediction']==c['prediction'] and r['task_score']==c['task_score']
                reused+=1
    df=pd.DataFrame(rows)
    from src.week11.reporting import summarize
    summary=summarize(df)
    assert summary.n_queries.eq(50).all()
    summary.to_csv(root/'audited_summary.csv',index=False)
    pd.DataFrame(flag_rows).to_csv(root/'flagged_queries.csv',index=False)
    empty=df[df.evidence_granularity!='NONE'].assign(empty=lambda x:x.selected_unit_ids.map(len).eq(0))
    empty.groupby(['dataset','reader','evidence_granularity','budget'],sort=False).agg(empty_rows=('empty','sum'),rows=('empty','size'),empty_rate=('empty','mean')).reset_index().to_csv(root/'empty_evidence_audit.csv',index=False)
    counts={flag:sum(flag in r['flags'].split(';') for r in flag_rows) for flag in ['empty_evidence_strict_prefix','invalid_verification_label','generation_cap_reached','unmapped_gold_annotation']}
    result=dict(phase=phase,timestamp_utc=datetime.now(timezone.utc).isoformat(),checks_passed=True,
        checksums_verified=len(rows),expected=len(expected),queries=len(cohort),counts_per_dataset=dict(Counter(r['dataset'] for r in rows)),
        counts_per_reader=dict(Counter(r['reader'] for r in rows)),Qwen_C_results_exactly_reused=reused,
        physical_generation_calls=int(df.physical_lm_calls.sum()),flags=counts,pending=status.get('pending',[]),
        core_code_hash=identity['code_hash'],config_hash=identity['config_hash'],status_sha256=file_hash(root/'status.json'),audit_driver_sha256=file_hash(Path(__file__)))
    atomic(root/'readiness_audit.json',result)
    text=[f'# Week 11 {phase} — audited pilot results',
        f"{len(rows):,} available logical rows completed; 50 queries per dataset. Pending components: {status.get('pending',[])}. This is the frozen development pilot, not full-dataset confirmation.",
        'Source/config/code identities, checksums, full action membership, budgets and same-reader/checkpoint HELP/HURT pairs passed. No scientific outputs were replaced.']
    if phase=='D':text+=['All readers reused the exact reviewed C evidence selections. All 5,550 Qwen predictions and scores exactly match C; these are reused observations, not new replications. Reader comparisons isolate the reader under fixed exposed evidence.']
    if phase=='E':text+=['C0 uses raw top-10-pool passages; C2 exposes up to eight exact query-ranked sentences with the same budgets. LLMLingua C1 remains pending. C3 is a QASPER/Mistral-only SARA specialist with its own adapted no-evidence baseline; it is not a portable text compressor and must be discussed separately. E has a different candidate pool width from C/D; compare compressor conditions within E for controlled conclusions.']
    pair_files=[p for p in root.glob('*.csv') if ('pairwise' in p.name or 'statistical_comparisons' in p.name)]
    pairs=pd.read_csv(pair_files[0]) if pair_files else pd.DataFrame()
    def interval(dataset,a,b):
        if pairs.empty:return None
        direct=pairs[(pairs.dataset==dataset)&(pairs.action_a==a)&(pairs.action_b==b)]
        if len(direct):return direct.iloc[0][['point','ci_low','ci_high']].to_dict()
        reverse=pairs[(pairs.dataset==dataset)&(pairs.action_a==b)&(pairs.action_b==a)]
        if len(reverse):
            r=reverse.iloc[0];return dict(point=-r.point,ci_low=-r.ci_high,ci_high=-r.ci_low)
        return None
    for dataset,g in summary.groupby('dataset'):
        text += [f'## {dataset}', 'Metric: '+('classification accuracy' if dataset=='SciFact' else 'answer F1')+'. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.']
        cols=['reader','reader_variant','evidence_granularity','selector','budget','compressor','task_score','delta_task_score','HELP','HURT','actual_evidence_tokens']
        displayed=[g[g.evidence_granularity=='NONE']]
        for _,h in g[g.evidence_granularity!='NONE'].groupby(['reader','reader_variant']):displayed.append(h.sort_values(['task_score','prompt_tokens'],ascending=[False,True]).head(3))
        text += [pd.concat(displayed)[cols].to_markdown(index=False,floatfmt='.6f')]
        intervals=[]
        for _,h in g[g.evidence_granularity!='NONE'].groupby(['reader','reader_variant']):
            best=h.sort_values(['task_score','prompt_tokens'],ascending=[False,True]).iloc[0]
            controls=g[(g.reader==best.reader)&(g.reader_variant==best.reader_variant)&(g.evidence_granularity=='NONE')]
            raw=g[(g.reader==best.reader)&(g.reader_variant==best.reader_variant)&(g.evidence_granularity=='PASSAGE')&(g.selector=='semantic')&(g.budget.astype(str)==str(best.budget))&(g.compressor=='C0')&(g.retriever==best.retriever)]
            for label,group in [('same-reader no evidence',controls),('matched-budget raw semantic passage',raw)]:
                for _,control in group.iterrows():
                    if control.action_id==best.action_id:continue
                    ci=interval(dataset,best.action_id,control.action_id)
                    if ci is not None:intervals.append(dict(reader=best.reader,variant=best.reader_variant,action=best.action_id,reference=label,**ci))
        if intervals:text+=['Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):',pd.DataFrame(intervals).to_markdown(index=False,floatfmt='.6f')]
    text+=['## Statistical interpretation',
        'Use paired 10,000-resample bootstrap intervals (seed 42): QASPER paper clusters, other datasets query/claim clusters. Saved comparisons include same-reader no-evidence and matched raw-passage controls; D also includes matched Mistral reader controls. A positive point estimate alone is insufficient for superiority. Intervals are nominal and exploratory; selecting best conditions on this cohort adds selection bias. No mixed raw QA/verification aggregate is valid.',
        'Paired result files: '+', '.join(p.name for p in pair_files)+'.',
        '## Flags', '\n'.join(f'- {name}: {count} logical rows.' for name,count in counts.items()),
        'Empty evidence follows the frozen strict-prefix budget rule; an oversized first unit is not silently skipped. Generation-cap hits require inspection, not deletion. Invalid SciFact outputs retain the predefined strict scoring. Gold-mapping gaps affect evidence-recall interpretation. Exact source support does not imply answer correctness.',
        f"Cost: {int(df.physical_lm_calls.sum()):,} physical calls for {len(rows):,} logical rows. Cache hits retain conceptual per-policy LM costs. Cached original timing fields are not measured end-to-end latency; that field remains null.",
        'Oracles are retrospective ceilings and do not establish a deployable controller. Specialist/public-reader surfaces are separated in the saved reports.',
        '## Complete condition summary',summary[cols+['dataset','prompt_tokens','generated_tokens']+(['invalid_label_rate'] if 'invalid_label_rate' in summary else [])].to_markdown(index=False,floatfmt='.6f')]
    (root/'INTERPRETATION.md').write_text('\n\n'.join(text)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['D','E','F','G'])
    audit(parser.parse_args().phase)
