import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src.week12.protocol import load,digest,immutable,ROOT
from src.week12.splits import check_disjoint,read_freeze,load_examples,assign_groups
from src.week12.training_v2 import condition,balancing,validate_rows,validation_metrics

def member(q,g=None):return dict(query_id=q,question_hash='hash-'+q,group=g or q,paper_id=g or q)
class CorrectedSplitTests(unittest.TestCase):
 def setUp(self):self.parts=dict(TRAIN=[member('a')],VAL=[member('b')],CONFIRM=[member('c')])
 def test_disjoint_all_three_roles(self):self.assertTrue(check_disjoint('QASPER',self.parts))
 def test_train_val_overlap_rejected(self):
  self.parts['VAL']=self.parts['TRAIN']
  with self.assertRaises(ValueError):check_disjoint('QASPER',self.parts)
 def test_train_confirm_overlap_rejected(self):
  self.parts['CONFIRM']=self.parts['TRAIN']
  with self.assertRaises(ValueError):check_disjoint('QASPER',self.parts)
 def test_val_confirm_overlap_rejected(self):
  self.parts['CONFIRM']=self.parts['VAL']
  with self.assertRaises(ValueError):check_disjoint('QASPER',self.parts)
 def test_qasper_paper_overlap_rejected(self):
  self.parts['VAL'][0]['group']='a'
  with self.assertRaises(ValueError):check_disjoint('QASPER',self.parts)
 def test_duplicate_question_with_different_id_rejected(self):
  self.parts['VAL'][0]['question_hash']='hash-a'
  with self.assertRaises(ValueError):check_disjoint('HotpotQA',self.parts)
 def test_group_assignment_is_deterministic_and_indivisible(self):
  xs=[member('a','g'),member('b','g'),member('c','h'),member('d','j')];a,b=assign_groups(xs,42,1);self.assertFalse({r['group'] for r in a}&{r['group'] for r in b});self.assertEqual((a,b),assign_groups(xs,42,1))
 def test_week12_loader_refuses_confirmation_before_sources(self):
  with patch('src.week12.splits.source_paths',side_effect=AssertionError('must not open sources')):
   for role in ['CONFIRM','TEST','confirmation']:
    with self.assertRaises(ValueError):load_examples(load(),role)
 def test_checkpoint_selection_refuses_train(self):
  with self.assertRaises(ValueError):validation_metrics(None,None,[dict(partition_role='TRAIN')],load())
 def test_uncapped_final_training_default(self):
  c=load();self.assertIsNone(c['training_queries_per_dataset']);self.assertEqual(c['training_data_policy'],'ALL_ELIGIBLE_TRAIN')
 def test_one_reproducible_condition_per_query(self):
  c=load();self.assertEqual(condition(c,'QASPER','q'),condition(c,'QASPER','q'));self.assertIn(condition(c,'QASPER','q'),[(m,b) for m in c['lora']['modes'] for b in c['budgets']])
 def test_checkpoint_val_condition_is_fixed(self):self.assertEqual(condition(load(),'SciFact','123','VAL'),('QBGF_FULL',256))
 def test_balanced_dataset_probability_not_dataset_size(self):
  rows=[dict(dataset=d) for d,n in [('a',5000),('b',500),('c',50)] for _ in range(n)];from src.week12.training_v2 import balanced_smoke_sampler
  indices,plan=balanced_smoke_sampler(rows)
  self.assertEqual(set(plan['sampling_probabilities'].values()),{1/3});self.assertEqual(len(indices),5550)
  for n in plan['planned_sample_counts'].values():self.assertLess(abs(n/5550-1/3),.025)
 def test_freeze_immutable(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'freeze.json';immutable(p,dict(ids=['a']));
   with self.assertRaises(ValueError):immutable(p,dict(ids=['b']))
 def test_final_freeze_required_before_week13_sources(self):
  from src.week12.confirmation import examples
  with patch('src.week12.confirmation.final_freeze',side_effect=RuntimeError('locked')),patch('src.week12.confirmation.source_paths',side_effect=AssertionError('must not open')):
   with self.assertRaises(RuntimeError):examples(load())
 def test_week13_only_materializes_frozen_confirm_ids(self):
  from src.week12.confirmation import examples
  split=dict(datasets={'HotpotQA':self.parts},notes={});r=dict(_id='c',question='q',answer='a',context=[['title',['sentence']]],supporting_facts=[]);other=dict(r,_id='b')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'dev.json';p.write_text(json.dumps([r,other]))
   with patch('src.week12.confirmation.final_freeze',return_value=({},split,'hash')),patch('src.week12.confirmation.source_paths',return_value={'HotpotQA_val':p}):
    xs=examples(load());self.assertEqual([x['query_id'] for x in xs],['c'])
 def test_corpus_must_use_all_train_no_cap(self):
  split=dict(datasets={'HotpotQA':dict(TRAIN=[member(str(i)) for i in range(501)],VAL=[member('v')],CONFIRM=[member('z')])})
  rows=[dict(dataset='HotpotQA',query_id=str(i),cluster_id=str(i),partition='train',partition_role='TRAIN') for i in range(501)]+[dict(dataset='HotpotQA',query_id='v',cluster_id='v',partition='validation',partition_role='VAL')]
  with patch('src.week12.training_v2.read_freeze',return_value=split):
   self.assertTrue(validate_rows(load(),rows))
   with self.assertRaises(ValueError):validate_rows(load(),rows[1:])
 def test_validation_cannot_be_gradient_row(self):
  split=dict(datasets={'HotpotQA':self.parts});rows=[dict(dataset='HotpotQA',query_id='a',cluster_id='a',partition='train',partition_role='VAL'),dict(dataset='HotpotQA',query_id='b',cluster_id='b',partition='validation',partition_role='VAL')]
  with patch('src.week12.training_v2.read_freeze',return_value=split):
   with self.assertRaises(ValueError):validate_rows(load(),rows)
 def test_real_frozen_roles_and_pilot_exclusions(self):
  f=read_freeze(load());self.assertFalse(f['notes']['QASPER']['fresh_confirmation_available']);self.assertFalse(f['datasets']['QASPER']['CONFIRM'])
  for d,p in f['datasets'].items():
   self.assertTrue(check_disjoint(d,p));self.assertFalse(set(f['historical_pilot_ids'][d])&{r['query_id'] for r in p['CONFIRM']})
  self.assertGreater(len(f['datasets']['HotpotQA']['TRAIN']),500)
 def test_qasper_test_path_blocked(self):
  from src.week12.protocol import guard
  with self.assertRaises(ValueError):guard('QASPER','confirmation','qasper-test.arrow')

class ValidationObjectiveSmoke(unittest.TestCase):
 def test_balanced_nll_and_task_metrics_on_synthetic_val_only(self):
  import torch
  from types import SimpleNamespace
  class Model(torch.nn.Module):
   device=torch.device('cpu')
   def forward(self,input_ids,**kwargs):return SimpleNamespace(loss=input_ids[0,0].float()/10)
   def generate(self,input_ids,**kwargs):return torch.cat([input_ids,torch.ones((1,1),dtype=torch.long)],dim=1)
  class Tok:
   eos_token_id=0
   def decode(self,*args,**kwargs):return 'SUPPORT'
  rows=[dict(partition_role='VAL',dataset=d,input_ids=[i,4],labels=[-100,4],prompt_ids=[i],references=['SUPPORT'],target='SUPPORT') for i,d in enumerate(['QASPER','HotpotQA','SciFact'],1)]
  m=Model();m.train();result=validation_metrics(m,Tok(),rows,load());self.assertAlmostEqual(result['eval_loss'],.2,places=6);self.assertEqual(result['data_role'],'VAL');self.assertEqual(set(result['validation_task_scores']),{'QASPER','HotpotQA','SciFact'});self.assertTrue(m.training)
