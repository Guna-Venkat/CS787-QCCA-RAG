import json,tempfile,unittest
from pathlib import Path
from src.week12.scheduling import full_coverage,CoverageLedger
from src.week12.execution import methods,budgets,cost_plan
from src.week12.protocol import load

class SchedulerCoreTests(unittest.TestCase):
 def rows(self):return [dict(dataset=d,query_id=str(i)) for d,n in [('A',7),('B',3),('C',2)] for i in range(n)]
 def test_full_permutation_no_replacement(self):
  rows=self.rows();ix,p=full_coverage(rows,5);self.assertEqual(sorted(ix),list(range(len(rows))));self.assertEqual(p['planned_repeats'],dict(A=0,B=0,C=0));self.assertEqual(ix,full_coverage(rows,5)[0])
 def test_equal_dataset_objective_mass(self):
  _,p=full_coverage(self.rows());self.assertEqual(set(p['objective_weight_mass'].values()),{4.});self.assertAlmostEqual(p['effective_loss_weights']['A'],12/21)
 def test_duplicate_train_identity_rejected(self):
  with self.assertRaises(ValueError):full_coverage([dict(dataset='A',query_id='q')]*2)
 def test_synthetic_full_coverage_optimizer_and_actual_ledger(self):
  import torch
  rows=self.rows();ix,p=full_coverage(rows,5);parameter=torch.nn.Parameter(torch.tensor(0.));opt=torch.optim.SGD([parameter],lr=.01)
  with tempfile.TemporaryDirectory() as d:
   log=CoverageLedger(Path(d)/'events.jsonl',[(r['dataset'],r['query_id']) for r in rows],p['effective_loss_weights'])
   for start in range(0,len(ix),5):
    window=ix[start:start+5]
    for i in window:
     r=rows[i];w=p['effective_loss_weights'][r['dataset']];loss=(parameter-float(i)).square()*w/len(window);loss.backward();log.forward([r['dataset']],[r['query_id']],[w])
    opt.step();opt.zero_grad();log.step(start//5+1)
   a=log.summary();self.assertEqual(a['physical_unique_examples_seen'],p['raw_eligible_examples']);self.assertEqual(a['physical_repeats'],p['planned_repeats']);self.assertEqual(a['applied_optimizer_steps'],3);self.assertFalse(any(a['missing_train_examples'].values()))
   self.assertEqual(a['optimizer_steps_with_dataset'],p['planned_optimizer_updates_with_dataset'])
   for ds,w in p['planned_optimizer_loss_coefficient_mass'].items():self.assertAlmostEqual(a['optimizer_loss_coefficient_mass'][ds],w)
 def test_partial_attempt_seen_but_not_an_optimizer_update(self):
  with tempfile.TemporaryDirectory() as d:
   log=CoverageLedger(Path(d)/'events.jsonl',[('A','q')],{'A':1.});log.forward(['A'],['q'],[1.]);a=log.summary();self.assertEqual(a['physical_unique_examples_seen']['A'],1);self.assertEqual(a['applied_optimizer_steps'],0)
   resumed=CoverageLedger(log.path,[('A','q')],{'A':1.});resumed.forward(['A'],['q'],[1.]);resumed.step(1);self.assertEqual(resumed.summary()['physical_repeats']['A'],1)
 def test_nontrain_forward_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   log=CoverageLedger(Path(d)/'events.jsonl',[('A','train')],{'A':1.})
   with self.assertRaises(ValueError):log.forward(['A'],['confirm'],[1.])
 def test_core_methods_and_secondary_explicit(self):
  self.assertEqual(methods('D'),['QBGF_FULL','QBGF_NO_QUERY','QBGF_SENTENCE_ONLY','QBGF_PASSAGE_ONLY']);self.assertNotIn('RAW_STRICT',methods('C'));self.assertIn('RAW_STRICT',methods('C',False));self.assertIn('QBGF_BM25',methods('D',False))
 def test_core_budgets_c2_only_registered_primary(self):
  c=load();self.assertEqual(budgets(c,'C','C2'),[256]);self.assertEqual(budgets(c,'C','C2',False),[256,1024]);self.assertEqual(budgets(c,'D','QBGF_NO_QUERY'),[256,1024])
 def test_cost_audit_exact_counts(self):
  p=cost_plan(load(),dict(QASPER=1005,HotpotQA=7155,SciFact=177));core=[r for r in p if r['CORE_FINAL']];self.assertEqual([r['logical_rows'] for r in core],[91707,66696]);self.assertEqual(sum(r['additional_call_upper_bound'] for r in core),133392)
 def test_secondary_path_is_distinct(self):
  from src.week12.runner import directory
  c=load();self.assertNotEqual(directory(c,'C'),directory(c,'C',core_final=False))
 def test_synthetic_core_surface_has_no_secondary_condition(self):
  from src.week12.runner import execute
  rows=execute('C',load(),smoke=True,limit=3);self.assertEqual(len(rows),33);self.assertFalse(any(r['method'] in ('RAW_STRICT','STRONG_SFR_RAG','BM25_QWEN_REFERENCE') for r in rows));self.assertFalse(any(r['method']=='C2' and r['budget']==1024 for r in rows))
 def test_optional_official_compressor_lock_valid_and_budget(self):
  from src.week12.compression_runtime import status
  v=status(load())
  if v['status']=='QUALIFIED':
   self.assertTrue(v['id'].startswith('microsoft/llmlingua-2-'));self.assertTrue(all(x['compressed_qwen_tokens']<=x['budget'] for x in v['synthetic_budget_checks']))

 def test_schedule_is_independent_of_corpus_storage_order(self):
  rows=self.rows();rev=list(reversed(rows));ix,p=full_coverage(rows);jx,q=full_coverage(rev)
  self.assertEqual([(rows[i]['dataset'],rows[i]['query_id']) for i in ix],[(rev[i]['dataset'],rev[i]['query_id']) for i in jx])

 def test_core_fusion_diagnostics_cover_both_budgets_and_all_datasets(self):
  from src.week12.statistics import registry,fusion_registry
  c=load();specs=fusion_registry(c);self.assertEqual(len(registry(c,True,True)),39);self.assertEqual(len(specs),12)
  self.assertEqual({s['right'] for s in specs},{'QBGF_SENTENCE_ONLY','QBGF_PASSAGE_ONLY'});self.assertEqual({s['budget'] for s in specs},{256,1024})
 def test_core_d_surface_includes_both_fusion_controls(self):
  from src.week12.runner import execute
  rows=execute('D',load(),smoke=True,limit=3);self.assertEqual(len(rows),24)
  for d in load()['datasets']:
   for m in methods('D'):
    self.assertEqual({r['budget'] for r in rows if r['dataset']==d and r['method']==m},{256,1024})
  self.assertFalse(any(r['method'] in ('QBGF_NO_MASS','QBGF_GREEDY','QBGF_BM25') for r in rows))
