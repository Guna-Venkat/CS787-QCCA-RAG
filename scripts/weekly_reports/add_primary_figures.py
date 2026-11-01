"""Primary-figure-only supplement, independently chosen by artifact identity, not legacy prose."""
from reportlib import *
def main():
 specs={2:[('results/week2/plots/05_token_f1_vs_context_tokens.png','X-axis is mean context tokens; y-axis token F1. The fixed-k frontier shows the quality/cost tradeoff; k10 is a separately trained RAG system, not an isolated compression ablation.')],3:[('results/week3/plots/plot5_epsilon_vs_mean_k.png','X-axis is retrospective oracle tolerance epsilon; y-axis mean selected k. Lower cost follows outcome-aware selection, not a trained inference-time policy. Human-score figures are deliberately excluded because raw ratings are blank.')],5:[('results/week5/content_allocator/figures/pareto_quality_vs_tokens.png','X-axis is recorded context-token cost; y-axis answer F1. Learned-policy, static and oracle references share the plot; oracle points are retrospective and token-reference conflicts prevent cross-week cost claims.')],6:[('results/week6/passage_routing/figures/w6_quality_vs_budget.png','X-axis is high-fidelity passage budget k; y-axis answer F1 by routing method. Semantic routing is strong at k4/k6; this does not prove a universal retriever winner.')],9:[('results/week9/w9h_final/v2/report/quality_cost.png','X-axis is recorded prompt/token cost; y-axis historical TEST answer F1. Distinct reader/checkpoint/interface systems remain whole-system comparisons; this consumed TEST cannot guide further tuning.')],11:[('results/week11/multigranular/v1/granularity_budget/figures/HotpotQA_task_score_quality_budget.png','X-axis is evidence budget; y-axis HotpotQA task score (answer F1). It shows granularity/selector-dependent budget behavior on50 pilot queries, not full-corpus retrieval or universal superiority.')]} 
 for w,figures in specs.items():
  out=BASE/f'week_{w:02d}';freeze=json.loads((out/'ReportFreeze.json').read_text());inv=pd.read_csv(out/'SourceInventory.csv');old=dict(freeze)
  for name in freeze['reports']:
   p=out/name;text=p.read_text()
   if '## Additional primary visualizations' in text:continue
   lines=['## Additional primary visualizations','']
   for path,caption in figures:
    assert (ROOT/path).exists();rel=os.path.relpath(ROOT/path,out)
    lines += ['![Recorded experimental figure]('+rel+')','',caption,'','Source: ['+path+']('+rel+').','']
    text=text.replace('# Existing Markdown Used as Evidence','- ['+path+']('+rel+')\n\n# Existing Markdown Used as Evidence')
   text=text.replace('# Primary Sources Used','\n'.join(lines)+'\n# Primary Sources Used');p.write_text(text)
  for path,caption in figures:
   if path not in freeze['sources']:
    freeze['sources'][path]=sha(ROOT/path);inv=pd.concat([inv,pd.DataFrame([dict(week=w,part='primary-figure supplement',artifact=path,type='png',status='inspected',what_it_can_establish='Visualization only; exact values sourced from existing CSVs',sha256=sha(ROOT/path))])],ignore_index=True)
  inv.to_csv(out/'SourceInventory.csv',index=False);freeze['reports']={n:sha(out/n) for n in freeze['reports']};freeze['figures']=freeze.get('figures',0)+len(figures);freeze['figure_supplement']=dict(primary_only=True,legacy_narrative_used=False)
  (out/'PreFigureSupplementFreeze.json').write_text(json.dumps(old,indent=2)+'\n');(out/'ReportFreeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
if __name__=='__main__':main()
