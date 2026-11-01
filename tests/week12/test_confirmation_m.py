"""No real confirmation example loader or model is invoked by this test suite."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src.week12.protocol import ROOT,load,Store,immutable,digest,sha
from src.week12_confirmation import freeze_guard as g
from src.week12_confirmation.confirmation_analysis import completed_rows,analyze,analyze_synthetic,tables
from src.week12_confirmation.synthetic import run,Reader

class MetadataTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cfg=load();cls.f,cls.s=g.verify(cls.cfg)
 def bad(self,f=None,s=None):
  with self.assertRaises(ValueError):g.validate_metadata(f or self.f,s or self.s)
 def test_freeze_sha_and_identical_copies(self):
  a=ROOT/'results/week12/W13_FINAL_METHOD_FREEZE.json';b=ROOT/'results/week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json';self.assertEqual(sha(a),g.FREEZE_SHA);self.assertEqual(a.read_bytes(),b.read_bytes())
 def test_wrong_freeze_sha(self):
  with patch.object(g,'FREEZE_SHA','wrong'):
   with self.assertRaises(ValueError):g.verify(self.cfg)
 def test_split_sha(self):self.assertEqual(self.s['split_hash'],g.SPLIT_SHA)
 def test_wrong_split(self):s=copy.deepcopy(self.s);s['split_hash']='wrong';self.bad(s=s)
 def test_registry(self):self.assertEqual([p[0] for p in self.f['confirmation_methods']],g.METHODS);self.assertEqual(len(g.registry()),12);self.assertEqual(sum(p['family']=='primary' for p in g.registry()),8)
 def test_budget_registry(self):self.assertEqual(self.f['budgets'],[256,1024])
 def test_wrong_budget(self):f=copy.deepcopy(self.f);f['budgets']=[256];self.bad(f=f)
 def test_exact_membership(self):self.assertTrue(g.validate_metadata(self.f,self.s))
 def test_qasper_zero(self):self.assertEqual(self.s['datasets']['QASPER']['CONFIRM'],[])
 def test_changed_cohort(self):s=copy.deepcopy(self.s);s['datasets']['HotpotQA']['CONFIRM'][0]['query_id']='changed';self.bad(s=s)
 def test_disjoint_ids(self):
  s=copy.deepcopy(self.s);s['datasets']['HotpotQA']['TRAIN'].append(s['datasets']['HotpotQA']['CONFIRM'][0]);self.bad(s=s)
 def test_disjoint_groups(self):
  s=copy.deepcopy(self.s);s['datasets']['HotpotQA']['TRAIN'][0]['group']=s['datasets']['HotpotQA']['CONFIRM'][0]['group'];self.bad(s=s)
 def test_extra_method(self):f=copy.deepcopy(self.f);f['confirmation_methods'].append(['C2','BASE_FP16']);self.bad(f=f)
 def test_missing_method(self):f=copy.deepcopy(self.f);f['confirmation_methods'].pop();self.bad(f=f)
 def test_wrong_model(self):
  c=copy.deepcopy(self.cfg);c['models']['Qwen']['revision']='wrong'
  with self.assertRaises(ValueError):g.verify(c)
 def test_wrong_prompt_hash(self):
  f=copy.deepcopy(self.f);f['prompt_source_hash']='wrong'
  with patch('src.week12.confirmation.final_freeze',return_value=(f,self.s,g.FREEZE_SHA)):
   with self.assertRaises(ValueError):g.verify(self.cfg)
 def test_wrong_parser_hash(self):
  f=copy.deepcopy(self.f);f['parser_hash']='wrong'
  with patch('src.week12.confirmation.final_freeze',return_value=(f,self.s,g.FREEZE_SHA)):
   with self.assertRaises(ValueError):g.verify(self.cfg)
 def test_default_safe_gate(self):
  with self.assertRaises(PermissionError):g.authorize()
 def test_partial_unlock_not_enough(self):
  with self.assertRaises(PermissionError):g.authorize(full=True,allow_labels=True,ack_sha=g.FREEZE_SHA)
 def test_scoring_default_inaccessible(self):
  with patch('src.week12_confirmation.confirmation_analysis.completed_rows',side_effect=AssertionError('must not read')):
   with self.assertRaises(PermissionError):analyze(self.cfg,None)
 def test_loader_default_inaccessible(self):
  from src.week12_confirmation.cohort_loader import load_cohort
  with patch('src.week12.confirmation.examples',side_effect=AssertionError('must not load')):
   with self.assertRaises(PermissionError):load_cohort(self.cfg,None)
 def test_attribution_cannot_change_policy(self):
  p=g.protocol(self.cfg);self.assertFalse(p['attribution_used']);self.assertEqual(p['methods'],g.METHODS);self.assertNotIn('attribution',json.dumps(p['registry']))
 def test_notebook_defaults(self):
  import nbformat
  n=nbformat.read(ROOT/'notebooks/week12/week12_M_final_confirmation.ipynb',4);nbformat.validate(n);s=n.cells[1].source
  for flag in ['RUN_FULL','RUN_SMOKE','ALLOW_CONFIRMATION_LABEL_ACCESS','RUN_CONFIRMATION_ANALYSIS']:
   self.assertIn(flag+' = False',s)
  self.assertIn('CONFIRM_FINAL_FREEZE_SHA256 = None',s)
  for cell in n.cells:
   if cell.cell_type=='code':ast.parse(cell.source)
 def test_old_results_readonly(self):
  from src.week12_confirmation import confirmation_runner as r
  self.assertNotIn('results/week13',Path(r.__file__).read_text());self.assertNotIn('src/week12_attribution',Path(r.__file__).read_text())

class SyntheticTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_completion_resume_report_and_checksums(self):
  before=Reader.calls;out,ident=run(self.out);calls=Reader.calls-before;self.assertGreater(calls,0)
  rows=completed_rows(out,ident);self.assertEqual(len(rows),20);self.assertTrue(all(r['synthetic'] for r in rows));self.assertEqual({r['method'] for r in rows},set(g.METHODS))
  paths=list((out/'rows').glob('*.json'));mtimes={p.name:p.stat().st_mtime_ns for p in paths}
  before=Reader.calls;run(self.out);self.assertEqual(Reader.calls,before);self.assertEqual(mtimes,{p.name:p.stat().st_mtime_ns for p in paths})
  frames=analyze_synthetic(out,ident);self.assertEqual(len(frames['contrasts']),12);self.assertIn('SYNTHETIC',(out/'week12_M_final_confirmation_report.md').read_text());self.assertTrue((out/'result_checksums.json').exists())
 def test_report_before_completion(self):
  with self.assertRaises(ValueError):completed_rows(self.out,{'synthetic':True})
 def test_overwrite_protection(self):
  immutable(self.out/'a.json',{'a':1})
  with self.assertRaises(ValueError):immutable(self.out/'a.json',{'a':2})
 def test_incompatible_store(self):
  Store(self.out/'s',{'model':'one'})
  with self.assertRaises(ValueError):Store(self.out/'s',{'model':'two'})
 def test_corrupt_result_fails(self):
  out,ident=run(self.out);p=next(p for p in (out/'rows').glob('*.json') if p.name!='identity.json');p.write_text('{}')
  with self.assertRaises(ValueError):completed_rows(out,ident)
 def test_synthetic_cannot_read_real_root(self):
  with self.assertRaises(PermissionError):analyze_synthetic(g.folder(),{'synthetic':True})
 def test_invalid_output_counts_in_macro(self):
  from src.week12.statistics import metric
  rows=[dict(gold=l,prediction='INVALID_OUTPUT',dataset='SciFact') for l in ['SUPPORT','REFUTE','NOT_ENOUGH_INFO']];self.assertEqual(metric(rows,'macro_f1'),0.)
if __name__=='__main__':unittest.main()
