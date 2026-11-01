"""Deterministic operational tests; never load a model or run a notebook."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from operations.week11 import continue_study as supervisor
from operations.week11 import review_generation as auditor
from src.week11.cache import Store, digest, atomic


class ContinuationTests(unittest.TestCase):
    def test_completion_requires_report_and_idle_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic(root/'status.json',dict(executable_complete=True))
            with patch.object(supervisor,'output',return_value=root),patch.object(supervisor,'owner_active',return_value=False),patch.object(supervisor,'writer_active',return_value=False):
                self.assertFalse(supervisor.complete({},'D'))
                (root/'REPORT.md').write_text('fixture');(root/'figures.json').write_text('[]')
                self.assertTrue(supervisor.complete({},'D'))
                with patch.object(supervisor,'writer_active',return_value=True):self.assertFalse(supervisor.complete({},'D'))
                with patch.object(supervisor,'owner_active',return_value=True):self.assertFalse(supervisor.complete({},'D'))

    def fixture(self, directory):
        root=Path(directory);cfg=dict(epsilon=.01,generation=dict(max_new_tokens=128),retrieval_width=20)
        code={'fixture':'fixed'};sources={'fixture':'dataset'}
        actions=[dict(action_id='base',reader='Qwen',granularity='NONE'),dict(action_id='text',reader='Qwen',granularity='PASSAGE')]
        cohort=[dict(dataset='HotpotQA',query_id=str(i)) for i in range(50)]
        for phase in ['C','D']:
            target=root/phase;target.mkdir()
            identity=dict(code_hash=digest(code),config_hash=digest(cfg),dataset_hash=digest(sources),phase=phase)
            atomic(target/'manifest.json',dict(identity=identity));atomic(target/'cohort.json',dict(selected=cohort))
            atomic(target/'status.json',dict(executable_complete=True,synthetic=False,expected=100,completed=100,pending=[]))
            store=Store(target/'rows',identity)
            for q in cohort:
                for a in actions:
                    r=dict(**q,action_id=a['action_id'],evidence_granularity=a['granularity'],reader='Qwen',reader_variant='PUBLIC_TEXT_READER',
                        code_hash=identity['code_hash'],config_hash=identity['config_hash'],synthetic=False,total_measured_latency=None,
                        budget=0 if a['granularity']=='NONE' else 1024,actual_evidence_tokens=0 if a['granularity']=='NONE' else 20,
                        task_score=0.,source_support=1,delta_task_score=0.,HELP=False,HURT=False,answer_changed=False,
                        prediction='fixture',selected_unit_ids=[] if a['granularity']=='NONE' else ['source'],generated_tokens=1,unmapped_gold=0,
                        selector='semantic',physical_lm_calls=0,selection_hash=a['action_id'],evidence_hash=a['action_id'],compressor='C0',
                        prompt_tokens=40,task_type='multihop_qa',strategy='FIXED',retriever='SFR',experiment_id=phase,passage_representation='P0')
                    store.put(['HotpotQA',q['query_id'],a['action_id']],r)
        return root,cfg,code,sources,actions

    def context(self,stack,root,cfg,code,sources,actions):
        for name,value in [('load',lambda:cfg),('gate',lambda *_:None),('output',lambda c,p:root/p),
                ('code_identity',lambda:code),('scientific_config',lambda *_:cfg),('source_identity',lambda _:sources),
                ('plan',lambda *_:actions),('pending_reason',lambda *_:None),('action',lambda *a,**k:actions[0])]:
            stack.enter_context(patch.object(auditor,name,value))

    def test_audit_checks_all_rows_and_exact_qwen_reuse(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root,cfg,code,sources,actions=self.fixture(directory);self.context(stack,root,cfg,code,sources,actions)
            result=auditor.audit('D')
            self.assertEqual(result['checksums_verified'],100)
            self.assertEqual(result['Qwen_C_results_exactly_reused'],100)
            self.assertTrue((root/'D/INTERPRETATION.md').exists())

    def test_corrupt_checkpoint_stops_audit(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root,cfg,code,sources,actions=self.fixture(directory);self.context(stack,root,cfg,code,sources,actions)
            path=next((root/'D/rows').glob('?'*64+'.json'))
            obj=json.loads(path.read_text());obj['row']['prediction']='altered';atomic(path,obj)
            with self.assertRaises(ValueError):auditor.audit('D')

    def test_valid_checksum_but_changed_qwen_answer_stops_audit(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root,cfg,code,sources,actions=self.fixture(directory);self.context(stack,root,cfg,code,sources,actions)
            path=next((root/'D/rows').glob('?'*64+'.json'))
            obj=json.loads(path.read_text());obj['row']['prediction']='altered';obj['checksum']=digest(obj['row']);atomic(path,obj)
            with self.assertRaises(AssertionError):auditor.audit('D')

    def supervisor_context(self,stack,directory,fail=False):
        root=Path(directory);monitor=root/'monitoring';monitor.mkdir()
        cfg=dict(output_root='results/week11/fixture',readers=['Qwen'],finalist_run_ids=['manually-reviewed'])
        calls=[]
        for phase in 'DEFGHI':
            (root/phase).mkdir();atomic(root/phase/'status.json',dict(pending=['LLMLingua-2'] if phase in 'DEG' else []))
        def reviewed(cfg,phase,*args):calls.append(phase)
        def audited(phase):
            if fail:raise ValueError('fixture audit failure')
            return dict(checks_passed=True)
        for name,value in [('ROOT',root),('MONITOR',monitor),('load',lambda:cfg),('code_identity',lambda:{'fixture':'code'}),
                ('scientific_config',lambda c,p:c),('file_hash',lambda p:'fixture'),('progress',lambda *a:None),
                ('owner_active',lambda p:False),('writer_active',lambda p:False),('complete',lambda c,p:True),
                ('output',lambda c,p:root/p),('audit',audited),('analysis_audit',lambda c,p:dict(checks_passed=True)),('review',reviewed)]:
            stack.enter_context(patch.object(supervisor,name,value))
        stack.enter_context(patch.object(supervisor.sys,'argv',['continue_study','--execute']))
        return monitor,calls

    def test_supervisor_sequence_without_any_notebook_or_model(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            monitor,calls=self.supervisor_context(stack,directory)
            with patch.object(supervisor.subprocess,'Popen',side_effect=AssertionError('No notebook may launch in this fixture')):
                supervisor.main()
            self.assertEqual(calls,list('DEFGHI'))
            self.assertEqual(json.loads((monitor/'continuation_state.json').read_text())['status'],'COMPLETE_AVAILABLE_STUDY')

    def test_supervisor_stops_before_next_phase_on_audit_failure(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            monitor,calls=self.supervisor_context(stack,directory,fail=True)
            with self.assertRaises(ValueError):supervisor.main()
            self.assertEqual(calls,[])
            self.assertEqual(json.loads((monitor/'continuation_state.json').read_text())['status'],'STOPPED_ERROR')


if __name__=='__main__':unittest.main()
