"""Load a verbatim historical leaf function without unrelated broken CLI imports."""
import ast
from functools import lru_cache
from ..final_h import protocol as p


@lru_cache(maxsize=1)
def bm25_prompt():
    path=p.ROOT/'src/week2/sweep_runner.py'
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_evaluation_prompt']
    p.require(len(nodes)==1,'Historical BM25 prompt function missing/ambiguous')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,nodes[0]],type_ignores=[]))
    from src.week7.w7e.prompt_builder import COMPRESS
    namespace=dict(COMPRESS=COMPRESS)
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['build_evaluation_prompt']
