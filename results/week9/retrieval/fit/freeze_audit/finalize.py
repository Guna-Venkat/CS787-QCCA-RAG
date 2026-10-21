"""Finalize the CPU-only audit and freeze the existing selection-rule winner."""
import json
from pathlib import Path
import sys
from contextlib import closing
import sqlite3
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from src.week9.provenance import ROOT,branch_guard,file_hash,source_hashes,digest
from src.week9.caching import atomic_json
from src.week9.retrieval_cache import read_rows
from src.week9.manifests import freeze
from src.week9.config import load_config,load_frozen
from src.week9.runner import dependencies,interventions,build_identity
from src.week9.data import load_records

branch_guard();folder=Path(__file__).parent;base=folder.parent
checks=json.loads((folder/'audit_checks.json').read_text());assert checks['status']=='PASS'
p=pd.read_csv(folder/'per_paper_chunk_statistics.csv');r=pd.read_csv(folder/'representation_chunk_statistics.csv').set_index('representation')
a=pd.read_csv(folder/'all_configuration_audit.csv');proposed=json.loads((folder/'proposed_freeze.json').read_text())
manifest=json.loads((base/'w9a_manifest.json').read_text());assert manifest['source_hashes']==source_hashes()
# Check the representative-paper partition assumption against every ORIG query.
by_paper=p[p.representation=='ORIG'].set_index('paper_id');by_paper.index=by_paper.index.astype(str)
orig_variations=[]
for row in read_rows(base/'w9a_chunk_manifest_compact.sqlite3',{'representation':'ORIG'}):
    lengths=[c['token_length'] for c in row['chunks']];ref=by_paper.loc[str(row['paper_id'])]
    if len(lengths)!=ref.chunks_per_paper or abs(np.mean(lengths)-ref.mean_chunk_tokens)>1e-9 or abs(np.median(lengths)-ref.median_chunk_tokens)>1e-9:
        orig_variations.append(row['question_id'])
assert not orig_variations,'ORIG paper partitions differ; recompute query-weighted saturation before freeze'
selected=proposed['settings'];filters={k:selected[k] for k in ['representation','retriever','retrieval_width']}
rows=list(read_rows(base/'w9a_retrieval_results_compact.sqlite3',filters,metrics_only=True))
annotated=sum(row['all_gold_evidence_recall'] is not None for row in rows)
short=sum(row['retrieval_actual_count']<4 for row in rows)
actual_mean=float(np.mean([row['retrieval_actual_count'] for row in rows]))
expected=a[(a.representation==filters['representation'])&(a.retriever==filters['retriever'])&(a.retrieval_width==filters['retrieval_width'])].iloc[0]
assert abs(actual_mean-expected.expected_mean_retrieval_actual_count)<1e-9
assert abs(short/len(rows)-expected.fraction_queries_fewer_than_4_chunks)<1e-9
best=float(a.all_gold_evidence_recall.max());eligible=a[a.all_gold_evidence_recall>=best-.003-1e-12]
assert len(eligible[np.isclose(eligible.retrieved_context_tokens,eligible.retrieved_context_tokens.min(),rtol=0,atol=1e-9)])==1
# Existing K_E oracle code already has the exact requested definitions and ties.
bcfg=load_config(ROOT/'configs/week9/w9b.yaml');assert bcfg['evidence_counts']==[0,1,2,4]
from src.week9.ke_surface import surface_analysis
# Static interface check only; no synthetic outcomes saved in result artifacts.
assert surface_analysis.__defaults__==((0,1,2,4),.01)
# A three-objective plot: recall vs tokens, density as color, 3-D frontier circled.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,6))
points=ax.scatter(a.retrieved_context_tokens,a.all_gold_evidence_recall,c=a.evidence_overlap_density,cmap='viridis',s=32,alpha=.8)
f=a[a.pareto_3d];ax.scatter(f.retrieved_context_tokens,f.all_gold_evidence_recall,facecolors='none',edgecolors='black',s=65,linewidths=.6,label='Non-dominated in recall/tokens/density')
for rep,method,width,offset in [('SEM-4','RRF',20,(-150,-28)),('SEM-4','RRF',40,(-30,-45)),('ORIG','RRF',20,(-80,-30))]:
    row=a[(a.representation==rep)&(a.retriever==method)&(a.retrieval_width==width)].iloc[0]
    ax.annotate(f'{rep}/{method}@{width}',(row.retrieved_context_tokens,row.all_gold_evidence_recall),xytext=offset,textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','lw':.5})
ax.scatter([expected.retrieved_context_tokens],[expected.all_gold_evidence_recall],marker='*',s=180,c='red',edgecolors='black',linewidths=.5,label='Frozen: FIXED-5/SEMANTIC@20',zorder=10)
fig.colorbar(points,ax=ax,label='Mean coarse evidence overlap density')
ax.set(xlabel='Mean retrieved MiniLM tokens per query',ylabel='All annotated evidence recall',title='W9-A FIT audit: retrieval tradeoffs (no answer generation)')
ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(folder/'pareto_frontier.png',dpi=180);fig.savefig(folder/'pareto_frontier.svg');plt.close(fig)

def point(rep,method='RRF',width=20):return a[(a.representation==rep)&(a.retriever==method)&(a.retrieval_width==width)].iloc[0]
x=point('SEM-4');x40=point('SEM-4',width=40);o=point('ORIG');o40=point('ORIG',width=40);fixed4=point('FIXED-4',width=40)
selected_point=point(filters['representation'],filters['retriever'],filters['retrieval_width'])
# Lock the canonical preregistered choice, not a retrospective SEM-4 override.
freeze_path=ROOT/'results/week9/frozen/w9a_frozen_config.json'
if freeze_path.exists():
    old=load_frozen(freeze_path,'a');assert old['settings']==selected
else:freeze(freeze_path,'a',selected,manifest,True)
frozen=load_frozen(freeze_path,'a')
# Validate W9-B prerequisites/protocol without opening a generator or SFR tensor.
deps=dependencies(bcfg,'b','fit');assert deps['a']['settings']==selected
specs=interventions('b',bcfg,deps)
assert [s['evidence_k'] for s in specs]==[0,1,2,4] and all(s['selector']=='DraftSemantic' for s in specs)
state=dict(status='PASS',frozen_file=str(freeze_path.relative_to(ROOT)),frozen_sha256=file_hash(freeze_path),
    selection_split='fit',selected=filters,best_global_recall=best,recall_tolerance=.003,
    unique_minimum_token_cost_within_tolerance=True,timing_tie_break_needed=False,
    source_hashes_unchanged=True,orig_partition_variations=orig_variations,
    recall_annotated_queries=annotated,all_queries=len(rows),queries_with_pool_fewer_than_4=short,
    w9b_prerequisites='PASS',w9b_selector='DraftSemantic',w9b_evidence_counts=[0,1,2,4],epsilon_oracle=.01,
    model_generations=0,full_w9b_run_launched=False)
atomic_json(folder/'freeze_decision.json',state)
lines=['# W9-A FIT freeze audit','', 'BRANCH = gagan. CPU-only saved-artifact audit; no model generations or confirmation-split selection.','',
    f'Complete grid: {checks["rows"]:,} rows, {checks["queries"]:,} queries, {checks["papers"]} papers. Recall averages over {annotated:,} queries with annotated evidence; missing evidence is not a successful retrieval.','',
    '## Decision','',f'Frozen **{filters["representation"]}/{filters["retriever"]}@{filters["retrieval_width"]}** using the existing preregistered global .003 recall tolerance, then retrieved-token cost. No post-hoc SEM-4 preference was substituted. The minimum eligible token cost is unique, so mixed historical/optimized timing measurements do not affect this decision.','',
    f'Global best recall: {best:.6f}. Selected recall: {selected_point.all_gold_evidence_recall:.6f}; retrieved tokens: {selected_point.retrieved_context_tokens:.1f}; coarse density: {selected_point.evidence_overlap_density:.4f}.','',
    '## Chunk size and saturation','', '| Representation | Pooled mean / median chunk tokens | Mean / median chunks per paper | Queries with whole pool at W20 |', '|---|---:|---:|---:|']
for rep in ['ORIG','FIXED-4','SEM-4','SEM-5','SEM-7','SEM-8','FIXED-5']:
    rr=r.loc[rep];ss=point(rep)
    lines.append(f'| {rep} | {rr.pooled_mean_chunk_tokens:.1f} / {rr.pooled_median_chunk_tokens:.1f} | {rr.mean_chunks_per_paper:.2f} / {rr.median_chunks_per_paper:.0f} | {100*ss.fraction_queries_entire_pool_retrieved:.1f}% |')
lines+=['','Statistics use one representative cached partition per paper/representation (the same paper deduplication principle used to match fixed sizes). All ORIG query partitions were checked for equal length/count statistics. SEM/FIXED boundaries are question-independent by construction. Query-weighted availability uses FIT paper query counts and agrees with the selected actual retrieval rows.','',
    'SEM-5/7/8 are large, few-chunk regimes: around 97–98% of queries exhaust the entire pool at W20. W40 adds little because few extra chunks remain, rather than proving better candidate discrimination. Retrieved-token totals include semantic overlap/repetition.','',
    '## The provisional SEM-4/RRF@20 point','',
    f'- Recall {x.all_gold_evidence_recall:.6f}, tokens {x.retrieved_context_tokens:.1f}, density {x.evidence_overlap_density:.4f}; it is on the three-objective Pareto frontier.',
    f'- Versus ORIG/RRF@20: +{100*(x.all_gold_evidence_recall-o.all_gold_evidence_recall):.3f} recall percentage points, but {100*(x.retrieved_context_tokens/o.retrieved_context_tokens-1):.1f}% more retrieved tokens.',
    f'- W20→W40 adds {100*(x40.all_gold_evidence_recall-x.all_gold_evidence_recall):.3f} recall points and {x40.retrieved_context_tokens-x.retrieved_context_tokens:.1f} tokens ({100*(x40.retrieved_context_tokens/x.retrieved_context_tokens-1):.1f}%). Mean actual candidates rise {x.expected_mean_retrieval_actual_count:.2f}→{x40.expected_mean_retrieval_actual_count:.2f}, not by 20. Already {100*x.fraction_queries_entire_pool_retrieved:.1f}% of queries exhaust the pool at W20.','',
    '## Length matching and semantic-boundary interpretation','',
    'The existing FIXED-i control matches the pooled median SEM-i chunk length, not the complete length distribution. SEM-4 has the same 205-token median as FIXED-4 but a much longer tail: pooled means 315.5 vs 201.3 tokens. A same-width gain therefore does not isolate semantic boundaries from length/budget.',
    f'At approximately matched retrieved-token budgets: SEM-4/RRF@20 uses {x.retrieved_context_tokens:.1f} tokens and recall {x.all_gold_evidence_recall:.6f}; FIXED-4/RRF@40 uses {fixed4.retrieved_context_tokens:.1f} and {fixed4.all_gold_evidence_recall:.6f}; ORIG/RRF@40 uses {o40.retrieved_context_tokens:.1f} and {o40.all_gold_evidence_recall:.6f}. The SEM-4 recall advantage is only {100*(x.all_gold_evidence_recall-fixed4.all_gold_evidence_recall):.3f} and {100*(x.all_gold_evidence_recall-o40.all_gold_evidence_recall):.3f} points respectively, with lower overlap density. These are near-budget comparisons, not exact causal length-matched experiments or significance claims.','',
    'All nominal median-matched SEM/FIXED pairs are saved in `length_matched_fixed_vs_semantic.csv`. All 171 configurations and the 3-D frontier are saved separately; plots encode density in color. Density is the project’s paragraph/word-overlap proxy, not exact evidence-token annotation or answer faithfulness.','',
    '## W9-B readiness and interpretation','',
    'W9-B already holds DraftSemantic fixed and evaluates K_E ∈ {0,1,2,4}. K=0 reuses the canonical initial draft. K>0 uses raw selected chunks at this stage; compact extraction is tested later in D/E. `surface_analysis()` implements the smallest exact-maximum K and the smallest K within .01 F1 of the maximum. No predictor is trained by B.',
    f'The frozen FIXED-5 chunks have median 1,022 tokens; a “unit” here is not a 205-token or extracted evidence item. {short}/{len(rows)} queries ({100*short/len(rows):.2f}%) have fewer than four candidates. Report actual counts and the full-four-candidate subgroup before interpreting nominal K=4 or concluding one compact unit suffices. The B notebook includes that analysis.',
    'A high K=2/4 epsilon-oracle mass would motivate investigating G, but does not replace G’s separate preregistered headroom/cost gate. B oracle results are retrospective ceilings, not deployable predictors.','',
    'The canonical frozen A artifact passes `load_frozen()` and B dependency/protocol checks. No B generation run was launched. Run manually:', '',
    '```bash','./sara_env/bin/python scripts/week9/run_w9b_ke_surface.py --split fit --config configs/week9/w9b.yaml --resume','```','',
    'Do not change A after B starts. A different compactness-constrained choice would be a new FIT-only amended experiment/output root, rather than an overwrite of this frozen decision.']
(folder/'W9A_FREEZE_AUDIT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(state,indent=2))
