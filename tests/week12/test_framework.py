import copy,itertools,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.week12 import algorithm as a
from src.week12.protocol import ROOT,Store,digest,load,authorize,guard
from src.week12.statistics import metric,paired,holm,behavioral,compression,pareto,registry
from src.week12.data import components
from src.week12.reader import parse_label,prompt_ids
from src.week11.budgets import FixtureTokenizer

class AlgorithmTests(unittest.TestCase):
 def setUp(self):
  self.t=FixtureTokenizer();self.doc={'document_id':'d','text':'Alpha red. Beta blue. Gamma green.'}
  self.p=dict(unit_id='p',passage_id='p',document_id='d',start_char=0,end_char=len(self.doc['text']),source_text=self.doc['text'],document_order=[0,0,0,0])
  self.s=[]
  for j,(x,y) in enumerate([(0,10),(11,21),(22,34)]):self.s.append(dict(unit_id=f's{j}',passage_id='p',document_id='d',start_char=x,end_char=y,source_text=self.doc['text'][x:y],document_order=[0,0,j,0]))
  self.ss={'s0':.1,'s1':.2,'s2':.9}
 def opts(self,m='QBGF_FULL'):return a.options([self.p],self.s,[.4],self.ss,self.t,m)
 def test_softmax_normalized(self):self.assertAlmostEqual(sum(a.softmax([1000,-1000,0])),1)
 def test_softmax_empty(self):self.assertEqual(a.softmax([]),[])
 def test_nonfinite_rejected(self):
  with self.assertRaises(ValueError):a.softmax([float('nan')])
 def test_cosine_normalized(self):self.assertTrue(np.allclose(a.cosine([2,0],[[1,0],[0,3]]),[1,0]))
 def test_mass_monotonic(self):
  m=[x['mass'] for x in self.opts()[0] if x['name'].startswith('SENT')];self.assertEqual(m,sorted(m));self.assertTrue(all(0<=v<=1+1e-12 for v in m))
 def test_mass_full_one(self):self.assertAlmostEqual(self.opts()[0][-1]['utility'],1.)
 def test_option_utility(self):
  o=self.opts()[0][1];self.assertAlmostEqual(o['utility'],a.softmax([.1,.2,.9])[2])
 def test_option_cost_matches_tokens(self):
  for o in self.opts()[0]:self.assertEqual(o['cost'],len(o['ids']))
 def test_no_duplicate_actual_k(self):self.assertEqual([o['actual_k'] for o in self.opts()[0] if o['name'].startswith('SENT')],[1,2,3])
 def test_source_exact(self):a.validate_spans(self.s,[self.doc])
 def test_source_mutation_rejected(self):
  x=copy.deepcopy(self.s);x[0]['source_text']='invented'
  with self.assertRaises(ValueError):a.validate_spans(x,[self.doc])
 def test_overlap_rejected(self):
  with self.assertRaises(ValueError):a.validate_spans([self.p,self.s[0]],[self.doc])
 def test_overlapping_pool_rejected(self):
  with self.assertRaises(ValueError):a.validate_pool([self.p,dict(self.p,unit_id='q')])
 def test_duplicate_pool_rejected(self):
  with self.assertRaises(ValueError):a.validate_pool([self.p,self.p])
 def test_document_provenance_not_text_dedup(self):a.validate_pool([self.p,dict(self.p,document_id='another',unit_id='q')])
 def test_source_order_restored(self):self.assertEqual([x['unit_id'] for x in a.serialize(self.s[::-1],self.t)['units']],['s0','s1','s2'])
 def test_selection_vs_presentation(self):
  o=self.opts()[0][2];self.assertEqual(o['selection_rank_ids'],['s2','s1']);self.assertEqual([u['unit_id'] for u in o['units']],['s1','s2'])
 def test_sentence_only(self):self.assertNotIn('FULL',[o['name'] for o in self.opts('QBGF_SENTENCE_ONLY')[0]])
 def test_passage_only(self):self.assertEqual([o['name'] for o in self.opts('QBGF_PASSAGE_ONLY')[0]],['DROP','FULL'])
 def test_no_query_score_independence(self):
  x=a.options([self.p],self.s,[-99],self.ss,self.t,'QBGF_NO_QUERY');y=a.options([self.p],self.s,[99],{k:-v for k,v in self.ss.items()},self.t,'QBGF_NO_QUERY');self.assertEqual(x,y)
 def test_no_mass_uniform_coverage(self):self.assertAlmostEqual(self.opts('QBGF_NO_MASS')[0][1]['utility'],1/3)
 def test_topk_highest_score(self):self.assertEqual(self.opts()[0][1]['units'][0]['unit_id'],'s2')
 def test_ties_deterministic(self):
  x=a.options([self.p],self.s,[0],{k:0 for k in self.ss},self.t);self.assertEqual(x[0][1]['units'][0]['unit_id'],'s0')
 def test_query_no_gold(self):
  p=copy.deepcopy(self.p);p.update(gold='secret',f1=1,answer='leak');self.assertEqual(self.opts()[0][1]['utility'],a.options([p],self.s,[.4],self.ss,self.t)[0][1]['utility'])
 def test_full_oversize_retains_sentences(self):
  x=a.solve(self.opts(),4);self.assertNotEqual(x[0]['name'],'DROP');self.assertNotEqual(x[0]['name'],'FULL')
 def test_zero_budget(self):self.assertEqual(a.solve(self.opts(),0)[0]['name'],'DROP')
 def test_negative_budget(self):
  with self.assertRaises(ValueError):a.solve(self.opts(),-1)
 def test_determinism(self):self.assertEqual(a.solve(self.opts(),8),a.solve(self.opts(),8))
 def test_actual_generator_ids_additive(self):
  class Merging:
   def encode(self,s,**kw):return list(range(len(s)//2))
  p=dict(self.p,unit_id='p2',passage_id='p2',document_id='d2',document_order=[1,0,0,0]);tok=Merging();x=a.serialize([self.p,p],tok);self.assertEqual(len(x['evidence_ids']),len(a.block([self.p],tok)['ids'])+len(a.block([p],tok)['ids']))
 def test_strict_historical_parity(self):
  from src.week11.budgets import allocate
  x=a.choose([self.p],self.s,[],[1],self.ss,self.t,4,'RAW_STRICT');y=allocate([self.p],4,{'Qwen':self.t});self.assertEqual(x['selected_unit_ids'],y['selected_unit_ids']);self.assertEqual(x['text'],y['text'])
 def test_raw_skip_skips(self):
  p2=dict(self.s[0],unit_id='p2',passage_id='p2',document_id='d2',document_order=[1,0,0,0]);x=a.choose([self.p,p2],[],[],[1,.5],{},self.t,5,'RAW_SKIP');self.assertEqual(x['selected_unit_ids'],['p2'])
 def test_budget_not_exceeded(self):
  for b in range(15):self.assertLessEqual(a.choose([self.p],self.s,[],[.4],self.ss,self.t,b,'QBGF_FULL')['evidence_tokens'],b)
 def test_one_option_per_passage(self):self.assertEqual(len(a.solve(self.opts(),10)),1)
 def test_dp_bruteforce_optimum(self):
  groups=[[{'cost':0,'utility':0},{'cost':3,'utility':5}],[{'cost':0,'utility':0},{'cost':2,'utility':3}],[{'cost':0,'utility':0},{'cost':2,'utility':3}]];best=max(sum(o['utility'] for o in x) for x in itertools.product(*groups) if sum(o['cost'] for o in x)<=4);self.assertEqual(sum(o['utility'] for o in a.solve(groups,4)),best)
 def test_greedy_is_separate(self):
  groups=[[{'cost':0,'utility':0},{'cost':3,'utility':5}],[{'cost':0,'utility':0},{'cost':2,'utility':3}],[{'cost':0,'utility':0},{'cost':2,'utility':3}]];self.assertEqual(sum(o['utility'] for o in a.solve(groups,4,True)),5)

class ProtocolTests(unittest.TestCase):
 def test_branch(self):guard()
 def test_other_branch_rejected(self):
  with patch('src.week12.protocol.subprocess.check_output',return_value='main\n'):
   with self.assertRaises(RuntimeError):guard()
 def test_test_block(self):
  with self.assertRaises(ValueError):guard('QASPER','test')
 def test_test_path_block(self):
  with self.assertRaises(ValueError):guard('QASPER','dev','QASPER_test.jsonl')
 def test_authorization_full_off(self):
  with self.assertRaises(ValueError):authorize(True,False,False)
 def test_authorization_full_smoke_conflict(self):
  with self.assertRaises(ValueError):authorize(True,True,True)
 def test_smoke_total_limit(self):
  with self.assertRaises(ValueError):authorize(False,True,False,4)
 def test_read_only_no_generation(self):
  from src.week12.runner import execute
  with patch('src.week12.runner.prepare',side_effect=AssertionError('generation attempted')):execute('A',load(),full=False,smoke=False)
 def test_store_resume(self):
  with tempfile.TemporaryDirectory() as d:
   s=Store(d,{'model':'pinned'});s.put(['q'],{'prediction':'a'});self.assertEqual(s.get(['q']),{'prediction':'a'})
 def test_store_overwrite_refused(self):
  with tempfile.TemporaryDirectory() as d:
   s=Store(d,{});s.put('q',{'a':1})
   with self.assertRaises(ValueError):s.put('q',{'a':2})
 def test_store_identity_refused(self):
  with tempfile.TemporaryDirectory() as d:
   Store(d,{'revision':'a'})
   with self.assertRaises(ValueError):Store(d,{'revision':'b'})
 def test_corruption_refused(self):
  with tempfile.TemporaryDirectory() as d:
   s=Store(d,{});s.put('q',{'a':1});p=Path(d)/(digest('q')+'.json');o=json.loads(p.read_text());o['row']['a']=3;p.write_text(json.dumps(o))
   with self.assertRaises(ValueError):s.get('q')
 def test_canonical_k_labels(self):
  from src.week12.canonical import prepare
  with self.assertRaises(ValueError):prepare(load(),[],'SARA_SFR_K6_HISTORICAL')
 def test_canonical_invalid_k(self):
  from src.week12.canonical import prepare
  c=load();c['sara']['canonical_k']=6
  with self.assertRaises(ValueError):prepare(c,[],'SARA_CANONICAL_BM25_K5_N10')
 def test_confirmation_execution_blocked(self):
  from src.week12.protocol import cohort
  with self.assertRaises(RuntimeError):cohort(load(),'confirmation')
 def test_transitive_document_groups(self):
  g=components([{'query_id':'a','split_documents':['x']},{'query_id':'b','split_documents':['x','y']},{'query_id':'c','split_documents':['y']}]);self.assertEqual(len(set(g.values())),1)
 def test_notebook_defaults(self):
  import nbformat,ast
  paths=list((ROOT/'notebooks/week12').glob('*.ipynb'));self.assertEqual(len(paths),15)
  for p in paths:
   n=nbformat.read(p,as_version=4);nbformat.validate(n);flags=n.cells[1].source
   if '_C_EXT_' in p.name:
    import subprocess
    flags=nbformat.reads(subprocess.check_output(['git','show',':'+str(p.relative_to(ROOT))],cwd=ROOT,text=True),as_version=4).cells[1].source
    self.assertIn('RUN_C_EXT_FULL=False',flags);self.assertIn('RUN_SYNTHETIC_SMOKE=True',flags);self.assertIn('RUN_SPECIALIST_QUALIFICATION=False',flags);self.assertIn('RESUME=True',flags);self.assertIn('OVERWRITE=False',flags)
   else:
    # User notebooks may currently be configured for a live full run. Check committed defaults.
    import subprocess
    try: saved=nbformat.reads(subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,text=True,stderr=subprocess.DEVNULL),as_version=4)
    except subprocess.CalledProcessError: saved=n
    flags=saved.cells[1].source
    self.assertIn('RUN_FULL = False',flags);self.assertTrue('RUN_SMOKE = True' in flags or (p.name.startswith(('week12_H_','week12_M_')) and 'RUN_SMOKE = False' in flags));self.assertIn('RESUME = True',flags);self.assertIn('OVERWRITE = False',flags)
   for cell in n.cells:
    if cell.cell_type=='code':ast.parse(cell.source)
 def test_no_automatic_final_selection(self):
  from src.week12.analysis import freeze_final
  self.assertEqual(freeze_final(load(),None,None)['status'],'REVIEW_REQUIRED')
 def test_registry_family_frozen(self):self.assertEqual(len(registry(load(),True,True)),39)
 def test_llmlingua_missing_safe(self):
  from src.week12.qualification import qualify_llm
  c=load()
  with tempfile.TemporaryDirectory() as d:
   c['output_root']=str(Path(d)/'outputs')
   with patch('src.week12.qualification.resources',return_value={'disk_free_GiB':1}),patch('src.week12.qualification.importlib.util.find_spec',return_value=None):self.assertEqual(qualify_llm(c)['status'],'NOT_QUALIFIED')

class MetricTests(unittest.TestCase):
 def test_label_trim_only(self):self.assertEqual(parse_label(' SUPPORT\n'),'SUPPORT')
 def test_label_case_invalid(self):self.assertEqual(parse_label('support'),'INVALID_OUTPUT')
 def test_label_explanation_invalid(self):self.assertEqual(parse_label('SUPPORT because...'),'INVALID_OUTPUT')
 def test_macro_f1_not_accuracy(self):
  rows=[{'gold':'SUPPORT','prediction':'SUPPORT','correct':1},{'gold':'REFUTE','prediction':'SUPPORT','correct':0},{'gold':'NOT_ENOUGH_INFO','prediction':'SUPPORT','correct':0}];self.assertAlmostEqual(metric(rows,'macro_f1'),1/6);self.assertAlmostEqual(metric(rows,'accuracy'),1/3)
 def test_invalid_is_false_negative(self):self.assertEqual(metric([{'gold':'SUPPORT','prediction':'INVALID_OUTPUT'}],'macro_f1'),0)
 def test_help_hurt_epsilon(self):self.assertTrue(behavioral(.51,.5,False,.02)['NEUTRAL']);self.assertTrue(behavioral(1,0,True)['HELP']);self.assertTrue(behavioral(0,1,True)['HURT'])
 def test_compression_reference_zero(self):self.assertIsNone(compression(0,0))
 def test_compression_ratio(self):self.assertEqual(compression(25,100),.75)
 def test_holm(self):self.assertTrue(np.allclose(holm([.01,.04,.03]),[.03,.06,.06]))
 def test_pareto_no_pooling(self):
  rs=[dict(dataset='a',reader='x',prompt_tokens=2,quality=.6),dict(dataset='a',reader='x',prompt_tokens=3,quality=.5),dict(dataset='b',reader='x',prompt_tokens=3,quality=.5)];self.assertEqual([r['pareto'] for r in pareto(rs)],[True,False,True])
 def test_bootstrap_macro_recomputed(self):
  a1=[dict(dataset='SciFact',query_id=str(i),cluster_id=str(i),gold=l,prediction=l) for i,l in enumerate(['SUPPORT','REFUTE','NOT_ENOUGH_INFO'])];b=copy.deepcopy(a1)
  for r in b:r['prediction']='SUPPORT'
  x=paired(a1,b,'macro_f1',n=200);self.assertAlmostEqual(x['point'],5/6);self.assertEqual(x['n_clusters'],3)
 def test_bootstrap_query_pairing(self):
  x=[dict(dataset='QASPER',query_id='a',cluster_id='paper',f1=1)]
  with self.assertRaises(ValueError):paired(x,[dict(x[0],query_id='b')],'f1',n=10)
 def test_bootstrap_paper_cluster(self):
  x=[dict(dataset='QASPER',query_id=str(i),cluster_id='paper',f1=1) for i in range(2)];y=[dict(r,f1=0) for r in x];self.assertEqual(paired(x,y,'f1',n=100)['n_clusters'],1)
 def test_qa_f1_em(self):
  from src.week11.evaluation import qa_pair
  self.assertEqual(qa_pair('the red signal','red signal'),(1.,1.))
 def test_smoke_lora_changes_only_adapters(self):
  from src.week12.lora import tiny_smoke
  with tempfile.TemporaryDirectory() as d:
   c=load();c['output_root']=d;r=tiny_smoke(c);self.assertTrue(r['base_unchanged'] and r['adapter_changed']);self.assertFalse(r['full_training_started'])
if __name__=='__main__':unittest.main()

class AdditionalSafetyTests(unittest.TestCase):
 def test_all_test_evaluation_blocked(self):
  with self.assertRaises(ValueError):guard('SciFact','test')
 def test_comparison_ids_unique(self):
  r=registry(load(),True,True);self.assertEqual(len(r),len({s['id'] for s in r}))
 def test_fixed_cohort_has_no_query_overlap(self):
  c=load(ROOT/'configs/week12/protocol_v1.yaml');p=ROOT/c['output_root']/'protocol/cohorts.json'
  if not p.exists():self.skipTest('Run notebook 0 first')
  x=json.loads(p.read_text())
  for d in c['datasets']:self.assertFalse({v['question_hash'] for v in x['development'][d]} & {v['question_hash'] for v in x['confirmation'][d]})
 def test_confirmation_reservation_is_not_evaluation_authority(self):
  c=load(ROOT/'configs/week12/protocol_v1.yaml');p=ROOT/c['output_root']/'protocol/W12_CONFIRMATION_FREEZE.json'
  if not p.exists():self.skipTest('Run notebook 0 first')
  x=json.loads(p.read_text());self.assertFalse(x['confirmation_evaluation_enabled']);self.assertEqual(x['history_status'],'REVIEW_REQUIRED');self.assertEqual(x['ids_hash'],digest(x['ids']))
 def test_pinned_encoder_reader(self):
  from src.week11.protocol import load as old
  c=load();w=old();self.assertEqual(c['models']['SFR'],w['models']['SFR']);self.assertEqual(c['models']['Qwen'],w['models']['Qwen'])
 def test_training_overlap_stops_before_model_load(self):
  from src.week12.lora import train
  with tempfile.TemporaryDirectory() as d:
   c=load();c['output_root']=d;p=Path(d)/'lora/corpus/rows.json';p.parent.mkdir(parents=True);p.write_text(json.dumps([dict(dataset='QASPER',cluster_id='paper',partition='train'),dict(dataset='QASPER',cluster_id='paper',partition='validation')]))
   with patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=RuntimeError('model must not load')):
    with self.assertRaises(AssertionError):train(c,explicit=True)
 def test_prompt_evidence_ids_not_retokenized(self):
  class Tok:
   def apply_chat_template(self,messages,**kw):return messages[0]['content']
   def encode(self,text,**kw):return [1]*len(text)
  evidence=[9191,9292];ids=prompt_ids(Tok(),'q','freeform_qa',evidence);self.assertEqual(ids.count(9191),1);self.assertEqual(ids.count(9292),1)
 def test_synthetic_limits_three_total(self):
  from src.week12.runner import examples
  self.assertEqual(len(examples(load(),True,3)),3)
 def test_readonly_w11_frozen_statuses(self):
  from src.week11.protocol import load as old,output,gate
  w=old()
  for phase in ['A','B','C','D','E','F','G','H','I']:gate(w,phase)
 def test_source_identity_changes_on_data_mutation(self):
  from src.week12.protocol import cached_sha
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_text('a');one=cached_sha(p);p.write_text('different');self.assertNotEqual(one,cached_sha(p))

class IntegrationSmokeTests(unittest.TestCase):
 def test_lora_2x2_synthetic_schema(self):
  from src.week12.runner import execute
  rows=execute('G',load(),smoke=True,limit=3)
  pairs={(r['method'],r['reader']) for r in rows};self.assertEqual(pairs,{('RAW_SKIP','BASE_NF4'),('RAW_SKIP','LORA_NF4'),('QBGF_FULL','BASE_NF4'),('QBGF_FULL','LORA_NF4')});self.assertTrue(all(r['synthetic'] for r in rows));self.assertEqual(len(rows),24)
 def test_exact_algorithm_is_not_a_trained_controller(self):
  import inspect
  self.assertNotIn('gold',inspect.signature(a.choose).parameters)
 def test_smoke_default_has_no_lora_checkpoint(self):
  from src.week12.runner import phase_methods
  self.assertEqual(phase_methods('G'),['RAW_SKIP','QBGF_FULL'])

class CanonicalPinTests(unittest.TestCase):
 def test_actual_canonical_base_pin(self):
  from src.week12.canonical import assert_base_pin
  c=load();assert_base_pin(c,ROOT/c['sara']['checkpoint'])
 def test_wrong_adapter_base_fails(self):
  from src.week12.canonical import assert_base_pin
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'adapter_config.json').write_text(json.dumps(dict(base_model_name_or_path='wrong/model')))
   with self.assertRaisesRegex(ValueError,'adapter base'):assert_base_pin(load(),p)

class TrainingQuestionGroupsTests(unittest.TestCase):
 def test_duplicate_normalized_questions_group_together(self):
  from src.week12.lora import training_groups
  rows=[dict(query_id='a',question='Who discovered X?',cluster_id='a'),dict(query_id='b',question='  WHO discovered   X? ',cluster_id='b'),dict(query_id='c',question='Different question?',cluster_id='c')]
  result=training_groups(rows,'HotpotQA');self.assertEqual(result[0]['cluster_id'],result[1]['cluster_id']);self.assertNotEqual(result[0]['cluster_id'],result[2]['cluster_id'])
