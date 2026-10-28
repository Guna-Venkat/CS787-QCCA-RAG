"""Run canonical frozen readers in owned processes; release all memory on close.

No model/prompt/decoding/scoring substitutions. IPC time is not misreported as
generation time or measured end-to-end latency. Only operational placement changes.
"""
import multiprocessing
import traceback
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_project():
    """Spawn may inherit a SARA-first path; pin project subpackages explicitly."""
    project = str(ROOT)
    sys.path[:] = [project] + [p for p in sys.path if p != project]
    package = importlib.import_module('src')
    source = str(ROOT/'src')
    # Preserve the separately namespaced SARA src.model extension, but resolve
    # week11/week10 from the project even if src was imported before this worker.
    package.__path__ = [source] + [p for p in package.__path__ if p != source]
    importlib.invalidate_caches()
    return dict(project_root=project,src_search_paths=list(package.__path__))


def _verify_module(module_name, relative):
    module=importlib.import_module(module_name)
    if Path(module.__file__).resolve() != (ROOT/relative).resolve():
        raise ImportError('Canonical reader module resolved outside the project: '+module_name)
    return str(Path(module.__file__).resolve())


class ReaderProcessError(RuntimeError):
    pass


def _worker(connection, kind, payload):
    reader = None
    try:
        bootstrap = _bootstrap_project()
        if kind == 'text':
            bootstrap['reader_module'] = _verify_module('src.week11.readers','src/week11/readers.py')
            from src.week11.readers import Reader
            reader = Reader(*payload)
        elif kind == 'sara':
            bootstrap['reader_module'] = _verify_module('src.week10.root_cause.runtime','src/week10/root_cause/runtime.py')
            from src.week10.root_cause.runtime import SaraReader
            reader = SaraReader(*payload)
        elif kind == 'fixture':
            reader = _Fixture(*payload)
        else:
            raise ValueError('Unknown canonical reader kind')
        connection.send(('ready', bootstrap))
        while True:
            method, args, kwargs = connection.recv()
            if method == 'close':
                reader.close();reader = None
                connection.send(('closed', None))
                return
            if method == 'tokenize':
                connection.send(('result', reader.tokenizer.encode(*args, **kwargs)))
                continue
            if method not in ('generate', 'generate_text'):
                raise ValueError('Unsupported reader method')
            connection.send(('result', getattr(reader, method)(*args, **kwargs)))
    except BaseException as exc:
        try:connection.send(('error', dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())))
        except (OSError, EOFError, BrokenPipeError):pass
    finally:
        if reader is not None:
            try:reader.close()
            except BaseException:pass
        connection.close()


class _ProcessReader:
    def __init__(self, kind, *payload):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, kind, payload))
        self.process.start();child.close()
        self.closed = False
        try:self.bootstrap = self._receive('ready')
        except BaseException:
            self.close();raise

    def _receive(self, expected):
        if not self.connection.poll(900):
            raise ReaderProcessError('Owned reader did not respond within 900 seconds')
        try:kind, value = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise ReaderProcessError('Owned reader exited without a result') from exc
        if kind == 'error':
            raise ReaderProcessError(f"{value['type']}: {value['message']}\n{value['traceback']}")
        if kind != expected:raise ReaderProcessError('Unexpected reader protocol response: '+kind)
        return value

    def generate(self, *args, **kwargs):
        return self._call('generate', args, kwargs)

    def generate_text(self, *args, **kwargs):
        return self._call('generate_text', args, kwargs)

    def _call(self, method, args, kwargs):
        if self.closed:raise ReaderProcessError('Reader is closed')
        self.connection.send((method, args, kwargs))
        return self._receive('result')

    def close(self):
        if self.closed:return
        self.closed = True
        try:
            if self.process.is_alive():
                try:
                    self.connection.send(('close', (), {}))
                    if self.connection.poll(20):self.connection.recv()
                except (EOFError, OSError, BrokenPipeError):pass
                self.process.join(20)
            if self.process.is_alive():
                self.process.terminate();self.process.join(10)
            if self.process.is_alive():
                self.process.kill();self.process.join(10)
        finally:self.connection.close()


class IsolatedTextReader(_ProcessReader):
    def __init__(self, cfg, name):super().__init__('text', cfg, name)


class IsolatedSaraReader(_ProcessReader):
    def __init__(self, cfg):
        super().__init__('sara', cfg)
        self.tokenizer = _RemoteTokenizer(self)


class _RemoteTokenizer:
    def __init__(self, owner):self.owner = owner
    def encode(self, *args, **kwargs):return self.owner._call('tokenize', args, kwargs)


def install():
    import src.week11.runner as runner
    import src.week10.root_cause.runtime as runtime
    runner.Reader = IsolatedTextReader
    # Spawned workers import fresh modules, so they use the canonical class;
    # only the orchestration process receives this lazy-import substitution.
    runtime.SaraReader = IsolatedSaraReader


class _Fixture:
    """Process/IPC lifecycle test only; never instantiated by a real reader."""
    def __init__(self, fail=False):
        if fail:raise ValueError('fixture initialization failure')
    def generate(self, question, evidence, task_type):
        return dict(prediction=question+'|'+evidence+'|'+task_type)
    def generate_text(self, question, evidence):return dict(content=question+'|'+evidence)
    def close(self):pass
