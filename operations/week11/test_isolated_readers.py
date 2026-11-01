"""Owned-process lifecycle tests; fixture worker loads no model or CUDA."""
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from operations.week11.isolated_readers import _ProcessReader, ReaderProcessError, _RemoteTokenizer


class IsolatedReaderTests(unittest.TestCase):
    def test_exact_request_result_and_owned_process_exit(self):
        reader=_ProcessReader('fixture')
        try:
            self.assertTrue(reader.process.is_alive())
            self.assertEqual(reader.generate('q','exact source','qa'),dict(prediction='q|exact source|qa'))
            self.assertEqual(reader.generate_text('q',''),dict(content='q|'))
        finally:reader.close()
        self.assertFalse(reader.process.is_alive())
        reader.close()
        with self.assertRaises(ReaderProcessError):reader.generate('q','x','qa')

    def test_worker_initialization_error_propagates(self):
        with self.assertRaisesRegex(ReaderProcessError,'fixture initialization failure'):
            _ProcessReader('fixture',True)

    def test_spawn_repairs_preimported_foreign_src_namespace(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            temp=Path(directory);foreign=temp/'foreign';(foreign/'src').mkdir(parents=True)
            (foreign/'src/__init__.py').write_text('# foreign fixture namespace\n')
            script=temp/'spawn_check.py'
            script.write_text(textwrap.dedent(f'''
                import sys,json
                sys.path.insert(0,{str(root)!r})
                from operations.week11.isolated_readers import _ProcessReader
                sys.path.insert(0,{str(foreign)!r})
                import src
                if __name__=='__main__':
                    reader=_ProcessReader('fixture')
                    try:
                        assert reader.bootstrap['src_search_paths'][0]=={str(root/'src')!r}
                        print(json.dumps(reader.generate('q','source','qa')))
                    finally:reader.close()
            '''))
            result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,check=True,timeout=30)
            self.assertEqual(json.loads(result.stdout),dict(prediction='q|source|qa'))

    def test_specialist_tokenizer_adapter_preserves_encode_arguments(self):
        class Owner:
            def _call(self, method, args, kwargs):
                self.request=(method,args,kwargs)
                return [1,2,3]
        owner=Owner()
        self.assertEqual(_RemoteTokenizer(owner).encode('source',add_special_tokens=False),[1,2,3])
        self.assertEqual(owner.request,('tokenize',('source',),dict(add_special_tokens=False)))

    def test_abrupt_worker_death_is_detected(self):
        reader=_ProcessReader('fixture')
        reader.process.terminate();reader.process.join(10)
        try:
            with self.assertRaises((ReaderProcessError,BrokenPipeError,ConnectionResetError)):
                reader.generate('q','x','qa')
        finally:reader.close()
        self.assertFalse(reader.process.is_alive())


if __name__=='__main__':unittest.main()
